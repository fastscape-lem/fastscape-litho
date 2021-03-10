"""Main module."""
import numpy as np
import numba as nb
import xarray as xr
import xsimlab as xs
import pandas as pd
import math
import fastscape
from fastscape.processes.erosion import TotalErosion
from fastscape.processes.channel import DifferentialStreamPowerChannelTD
from fastscape.processes.hillslope import DifferentialLinearDiffusion
from fastscape.processes.grid import UniformRectilinearGrid2D
import xsimlab as xs

from fastscape.processes.boundary import BorderBoundary
from fastscape.processes.channel import (StreamPowerChannel,
                                 DifferentialStreamPowerChannelTD)
from fastscape.processes.context import FastscapelibContext
from fastscape.processes.flow import DrainageArea, SingleFlowRouter, MultipleFlowRouter, FlowRouter, FlowAccumulator
from fastscape.processes.erosion import TotalErosion
from fastscape.processes.grid import RasterGrid2D
from fastscape.processes.hillslope import LinearDiffusion, DifferentialLinearDiffusion
from fastscape.processes.initial import (BareRockSurface,
                                 Escarpment,
                                 FlatSurface,
                                 NoErosionHistory)
from fastscape.processes.main import (Bedrock,
                              StratigraphicHorizons,
                              SurfaceTopography,
                              SurfaceToErode,
                              TerrainDerivatives,
                              TotalVerticalMotion,
                              UniformSedimentLayer)
from fastscape.processes.marine import MarineSedimentTransport, Sea
from fastscape.processes.tectonics import (BlockUplift,
                                   SurfaceAfterTectonics,
                                   TectonicForcing,
                                   TwoBlocksUplift)
from ._looper import iterasator
import fastscape_litho._helper as fhl



##############################################################
##############################################################
####################### Processes ############################
##############################################################
##############################################################

@nb.njit()
def calculate_indices(cumulative_height, z, dz, nz, oz, indices, labelmatrix ):
  '''
    Calculates the indices from te 3D matrix functio of the depth
  '''
  for i in range(indices.shape[0]):
    for j in range(indices.shape[1]):
      # if(cumulative_height.shape[0] == 0):
      #   this_id = 0
      #   continue
      this_id = math.ceil((cumulative_height[i,j] - oz)/dz)
      if(this_id >= nz):
        indices[i,j] = -1
      else:
        indices[i,j] = labelmatrix[i,j,this_id]




@xs.process
class Label3D:
  '''
    base calss for the 3D label.
  '''

  dz = xs.variable(default = 50, description = 'Depth resolution of the 3D matrix of labels')
  origin_z = xs.variable(default = 0, description = 'original elevation of the grid. Leave to 0')
  z = xs.index(dims='z', description='Matrix depth coordinate')
  labelmatrix = xs.variable(dims = ('y','x','z'))
  cumulative_height = xs.foreign(TotalErosion, "cumulative_height")
  shape = xs.foreign(UniformRectilinearGrid2D,"shape")
  indices = xs.variable(
    intent = 'out',
    dims=[(), ('y', 'x')],
    description='indices of bedrock intersection'
  )

  runoff = xs.variable(
    intent = 'out',
    dims=[(), ('y', 'x')],
    description='surface runoff (source term) per area unit'
  )

  precipitations = xs.variable(
    intent = 'in',
    dims=[(), ('y', 'x')],
    description='surface runoff (source term) per area unit',
    default = 1
  )

  infiltration_lab = xs.variable(dims = 'n_labels', description = 'infiltration values for each label')


  def initialize(self):
    self.nz = self.labelmatrix.shape[2]
    self.z = np.linspace(-1 * self.origin_z, -1 * self.origin_z + self.nz * self.dz + self.dz , self.nz)

    self.indices = np.full(self.shape, -1, dtype = np.int32)
    if(isinstance(self.precipitations, (list, tuple, np.ndarray)) == False):
      self.precipitations = np.full(self.shape, self.precipitations, dtype = np.float64)

  def run_step(self):

    if(isinstance(self.cumulative_height,np.ndarray) ==  False):
      self.indices = np.zeros(self.shape, dtype = np.int32)
    else:
      calculate_indices(self.cumulative_height, self.z, self.dz, self.nz, self.origin_z, self.indices, self.labelmatrix )

    self.runoff = self.precipitations - self.infiltration_lab[self.indices]
    # print(np.unique(self.runoff))



@xs.process
class Label3DFull(Label3D):

  Kr_lab = xs.variable(dims = 'n_labels', description = 'Kr value for each label')
  Kdr_lab = xs.variable(dims = 'n_labels', description = 'Kdr value for each label')
  Kds_lab = xs.variable(dims = 'n_labels', description = 'Kds value for each label')
  # infiltration_lab = xs.variable(dims = 'n_labels', description = 'infiltration value for each label')

  k_coef_bedrock = xs.variable(intent = 'out',
    dims=[(), ('y', 'x')],
    description='bedrock channel incision coefficient'
  )
  diffusivity_bedrock = xs.variable(
    intent = 'out',
    dims=[(), ('y', 'x')],
    description='bedrock diffusivity'
  )
  diffusivity_soil = xs.variable(
    intent = 'out',
    dims=[(), ('y', 'x')],
    description='soil (sediment) diffusivity'
  )

  def run_step(self):

    super(Label3DFull, self).run_step()

    self.k_coef_bedrock = self.Kr_lab[self.indices]
    self.diffusivity_bedrock = self.Kdr_lab[self.indices]
    self.diffusivity_soil = self.Kds_lab[self.indices]


@xs.process
class Label3DSPL(Label3D):
  k_lab = xs.variable(dims = 'n_labels', description = 'K value for each label')
  k_coef = xs.variable(intent = 'out',
    dims=[(), ('y', 'x')],
    description='bedrock channel incision coefficient'
  )
  def run_step(self):

    super(Label3DSPL, self).run_step()

    self.k_coef = self.k_lab[self.indices]


@xs.process
class DifferentialStreamPowerChannelTDForeign(DifferentialStreamPowerChannelTD):
  k_coef_bedrock = xs.foreign(Label3DFull, 'k_coef_bedrock')

@xs.process
class DifferentialLinearDiffusionForeign(DifferentialLinearDiffusion):
  diffusivity_bedrock = xs.foreign(Label3DFull,'diffusivity_bedrock')
  diffusivity_soil = xs.foreign(Label3DFull,'diffusivity_soil')

@xs.process
class StreamPowerChannelForeign(StreamPowerChannel):
  k_coef = xs.foreign(Label3DSPL,"k_coef")

@xs.process
class FlowAccumulatorForeign(FlowAccumulator):
    """Accumulate the flow from upstream to downstream."""

    runoff = xs.foreign(Label3D,"runoff")

    shape = xs.foreign(UniformRectilinearGrid2D, 'shape')
    cell_area = xs.foreign(UniformRectilinearGrid2D, 'cell_area')
    stack = xs.foreign(FlowRouter, 'stack')
    nb_receivers = xs.foreign(FlowRouter, 'nb_receivers')
    receivers = xs.foreign(FlowRouter, 'receivers')
    weights = xs.foreign(FlowRouter, 'weights')

    flowacc = xs.variable(
        dims=('y', 'x'),
        intent='out',
        description='flow accumulation from up to downstream'
    )

    def run_step(self):
        field = (self.runoff * self.cell_area).flatten()
        # print("here1")
        if self.receivers.ndim == 1:
            fhl._flow_accumulate_sd(field, self.stack, self.receivers)

        else:
            fhl._flow_accumulate_mfd(field, self.stack, self.nb_receivers,
                                 self.receivers, self.weights)

        self.flowacc = field.reshape(self.shape)
        self.flowacc[self.flowacc <= 0] = 1
        # print("here2:", self.flowacc)


@xs.process
class FlexureLabel(fastscape.processes.Flexure):
  """Flexural isostatic effect of both erosion and tectonic
  forcing.

  """
  lithos_density = xs.variable(
      dims=[(), ('y', 'x')],
      description='lithospheric rock density (in this case, given)',
      intent = 'out'
  )

  rho_lab = xs.variable(dims = 'n_labels', description = 'Density value for each label')

  indices = xs.foreign(Label3D, "indices")


  def run_step(self):

    self.lithos_density = self.rho_lab[self.indices]
    super(FlexureLabel, self).run_step()




##############################################################
##############################################################
####################### Topographic Analysis #################
##############################################################
##############################################################



@xs.process
class Quicksn:
    
  # UniformRectilinearGrid2D
  dx = xs.foreign(UniformRectilinearGrid2D, 'dx')
  dy = xs.foreign(UniformRectilinearGrid2D, 'dy')

  nx = xs.foreign(UniformRectilinearGrid2D, 'nx')
  ny = xs.foreign(UniformRectilinearGrid2D, 'ny')

  x = xs.foreign(UniformRectilinearGrid2D, 'x')
  y = xs.foreign(UniformRectilinearGrid2D, 'y')

  fs_context = xs.foreign(FastscapelibContext, 'context')

  # FlowRouter
  elevation = xs.foreign(SurfaceToErode, 'elevation')
  stack = xs.foreign(FlowRouter, 'stack')
  nb_receivers = xs.foreign(FlowRouter, 'nb_receivers')
  receivers = xs.foreign(FlowRouter, 'receivers')
  lengths = xs.foreign(FlowRouter, 'lengths')
  weights = xs.foreign(FlowRouter, 'weights')
  nb_donors = xs.foreign(FlowRouter, 'nb_donors')
  donors = xs.foreign(FlowRouter, 'donors')
  boundary_conditions = xs.foreign(BorderBoundary, "status")

  flowacc = xs.foreign(FlowAccumulator, 'flowacc')

  theta_chi = xs.variable(intent = 'in', description = "Theta for chi calculation")
  A_0_chi = xs.variable(intent = 'in', description = "Reference drainage area for chi calculations")
  minAcc = xs.variable(intent = 'in', description = "minimum A/Q to do the analysis")
  output_prefix = xs.variable(intent = 'in', default = "TA_output")
  main_divide_contrast_distance = xs.variable(intent = 'in', default = 5000)
  thetas_default = np.arange(0.05,1,0.05)
  # main_divide_cross_theta_list = xs.variable(intent = 'in', dims= ((),'n_theta'), default = 5)

  internal_chi = xs.variable(intent = 'out',
    dims=('nodes'),
    description='chi coordinate'
    )

  chi = xs.on_demand(
    dims=('y', 'x'),
    description='chi coordinate'
    )

  ksn = xs.on_demand(
    dims=('y', 'x'),
    description='Local k_sn index (dz/dchi)'
    )

  ksnSA = xs.on_demand(
    dims=('y', 'x'),
    description='Local k_sn index (ksn = S A ^ theta)'
    )

  main_divide = xs.on_demand(
    dims=('y', 'x'),
    description = 'Label the areas draining to each fixed_value edges. Steepest Descent approximation.'
  )

  main_divide_cross_theta = xs.on_demand(
    dims=('n_theta'),
    description = 'Label the areas draining to each fixed_value edges. Steepest Descent approximation.'
  )

  main_drainage_divides_migration_index = xs.on_demand(description = "Represents the main drainage divide variations at each time step")

  def initialize(self):
    # getting boundary conditions right for the iteratool
    self.node_type = np.ones((self.ny,self.nx), dtype = np.int16)
    #Was drainage divide is an intermediate checker
    self.was_DD = np.zeros((self.ny,self.nx), dtype = np.bool)

    #left, right, top, bottom boundary conditions
    if(self.boundary_conditions[0] == "fixed_value"):
      self.node_type[:,0] = 0
    else:
      self.node_type[:,0] = 2
    if(self.boundary_conditions[1] == "fixed_value"):
      self.node_type[:,-1] = 0
    else:
      self.node_type[:,-1] = 2
    if(self.boundary_conditions[2] == "fixed_value"):
      self.node_type[0,:] = 0
    else:
      self.node_type[0,:] = 2
    if(self.boundary_conditions[3] == "fixed_value"):
      self.node_type[-1,:] = 0
    else:
      self.node_type[-1,:] = 2

    # getting the iteratoolto work
    self.iteratool = iterasator(self.node_type.ravel(),self.nx,self.ny, self.dx, self.dy)


  @xs.runtime(args = ["step_end", "step"])
  def run_step(self,step_end,step):
    self.step_end = str(step_end)
    self.step = str(step)

    # Just checking if self.is_multiple_flow
    self.is_multiple_flow = True if self.receivers.ndim > 1 else False
    if(self.is_multiple_flow == False):
      self.internal_chi = fhl.chiculation_SF(self.stack, self.receivers, self.nb_donors, self.donors, self.lengths, 
        self.elevation.ravel(), self.flowacc.ravel(), self.A_0_chi, self.theta_chi, self.minAcc).reshape(self.ny,self.nx)
    else:
      self.internal_chi = fhl.chiculation_MF(self.stack, self.receivers, self.nb_receivers, self.lengths, self.weights, 
        self.elevation, self.flowacc.ravel(), self.A_0_chi, self.theta_chi, self.minAcc)

  def finalize_step(self):

    basins = fhl.basination_SF(self.fs_context["stack"].astype('int') - 1, self.fs_context["rec"].astype('int') - 1)
    self.was_DD = fhl.is_draiange_divide_SF(basins, self.iteratool).reshape(self.ny, self.nx)




  def get_slope(self):
    if(self.is_multiple_flow):
      return fhl.slope_MF(self.elevation.ravel(), self.receivers, self.lengths, self.nb_receivers, self.weights)
    else:
      return fhl.slope_SF(self.elevation.ravel(), self.receivers, self.lengths)

  @chi.compute
  def _chi(self):
    return self.internal_chi.reshape(self.ny,self.nx)
    

  @ksn.compute
  def _ksn(self):
    if(self.is_multiple_flow == False):
      return fhl.ksn_calculation_SF(self.elevation.ravel(), self.internal_chi.ravel(), self.receivers, self.stack).reshape(self.ny,self.nx)
    else:
      return fhl.ksn_calculation_MF(self.elevation.ravel(), self.internal_chi.ravel(), self.receivers,self.nb_receivers, self.weights, self.stack).reshape(self.ny,self.nx)

  @ksnSA.compute
  def _ksnSA(self):
    ret = np.power(self.flowacc,self.theta_chi) * self.get_slope().reshape(self.ny,self.nx)
    ret[self.flowacc<self.minAcc] = 0
    return ret





  @main_drainage_divides_migration_index.compute
  def _main_drainage_divides_migration_index(self):
    sst = self.fs_context["stack"].astype('int') - 1
    # print(sst)
    # print(self.receivers)
    basins = fhl.basination_SF(sst, self.fs_context["rec"].astype('int') - 1)
    is_DD = fhl.is_draiange_divide_SF(basins, self.iteratool).reshape(self.ny, self.nx)
    return is_DD[(is_DD == 1) & (self.was_DD == 0) ].shape[0]


  @main_divide.compute
  def _main_divide(self):
    # First labelling the outlets
    main_divide = np.zeros_like(self.elevation,dtype = np.int16)
    #left, right, top, bottom boundary conditions
    if(self.boundary_conditions[0] == "fixed_value"):
      main_divide[:,0] = 1
    if(self.boundary_conditions[1] == "fixed_value"):
      main_divide[:,-1] = 2
    if(self.boundary_conditions[2] == "fixed_value"):
      main_divide[0,:] = 3
    if(self.boundary_conditions[3] == "fixed_value"):
      main_divide[-1,:] = 4

    main_divide = main_divide.ravel()

    fhl.extract_main_regions(self.fs_context["stack"].astype('int') - 1,
        main_divide, self.fs_context["rec"].astype('int') - 1)


    output = fhl.chi_contrast_across_divides(main_divide, self.internal_chi.ravel(), self.iteratool, self.main_divide_contrast_distance )
    df = pd.DataFrame({"X": output[0], "Y": output[1], "chi_med_zone": output[2], "chi_med_others": output[3],
      "chi_max_zone": output[4], "chi_max_others": output[5], "zone": output[6].astype(np.int)})
    
    tstep = str(self.step)
    while(len(tstep)<6):
      tstep = "0" +  tstep

    df.to_csv(self.output_prefix + "%s_mainDDinfo.csv"%(tstep), index = False)

    return main_divide.reshape(self.ny,self.nx)

  @main_divide_cross_theta.compute
  def _main_divide_cross_theta(self):

    dfs = []
    # First labelling the outlets
    main_divide = np.zeros_like(self.elevation,dtype = np.int16)
    #left, right, top, bottom boundary conditions
    if(self.boundary_conditions[0] == "fixed_value"):
      main_divide[:,0] = 1
    if(self.boundary_conditions[1] == "fixed_value"):
      main_divide[:,-1] = 2
    if(self.boundary_conditions[2] == "fixed_value"):
      main_divide[0,:] = 3
    if(self.boundary_conditions[3] == "fixed_value"):
      main_divide[-1,:] = 4

    main_divide = main_divide.ravel()

    fhl.extract_main_regions(self.fs_context["stack"].astype('int') - 1,
        main_divide, self.fs_context["rec"].astype('int') - 1)

    thetas = np.arange(0.05,1,0.05)
    dfs = []
    for theta in thetas:

      this_chi = fhl.chiculation_SF(self.fs_context["stack"].astype('int') - 1, 
        self.fs_context["rec"].astype('int') - 1, self.fs_context["ndon"].astype('int'),
         self.fs_context["don"].astype('int') - 1, self.fs_context["length"], 
          self.elevation.ravel(), self.flowacc.ravel(), self.A_0_chi, theta, self.minAcc)



      output = fhl.chi_contrast_across_divides(main_divide, this_chi.ravel(), self.iteratool, self.main_divide_contrast_distance )
      dfs.append(pd.DataFrame({"X": output[0], "Y": output[1], "chi_med_zone": output[2], "chi_med_others": output[3],
        "chi_max_zone": output[4], "chi_max_others": output[5], "zone": output[6].astype(np.int)}))

    tds = xr.concat([df.to_xarray() for df in dfs], dim="thetas")
    tds["thetas"] = thetas

    tstep = str(self.step)
    while(len(tstep)<6):
      tstep = "0" +  tstep

    tds.to_zarr(self.output_prefix + "%s_mainDDinfo.zarr"%(tstep))

    return thetas



  # UniformRectilinearGrid2D

##############################################################
##############################################################
####################### Models ###############################
##############################################################
##############################################################




sediment_model_label3D = fastscape.models.basic_model.update_processes({
    'bedrock': Bedrock,
    'active_layer': UniformSedimentLayer,
    'init_bedrock': BareRockSurface,
    'flow': MultipleFlowRouter,
    'spl': DifferentialStreamPowerChannelTDForeign,
    'diffusion': DifferentialLinearDiffusionForeign,
    'label': Label3DFull,
    'drainage': FlowAccumulatorForeign
})


sediment_model_label3D_SPL = fastscape.models.basic_model.update_processes({
    'bedrock': Bedrock,
    'active_layer': UniformSedimentLayer,
    'init_bedrock': BareRockSurface,
    'flow': MultipleFlowRouter,
    'spl': StreamPowerChannelForeign,
    'label': Label3DSPL,
    'drainage': FlowAccumulatorForeign
}).drop_processes({'diffusion'})

















































# End of file