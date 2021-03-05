"""Main module."""
import numpy as np
import numba as nb
import xarray as xr
import xsimlab as xs
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
import fastscape_litho._helper as fhl.



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


  def initialize(self):
    self.nz = self.labelmatrix.shape[2]
    self.z = np.linspace(-1 * self.origin_z, -1 * self.origin_z + self.nz * self.dz + self.dz , self.nz)

    self.indices = np.full(self.shape, -1, dtype = np.int32)

  def run_step(self):
    if(isinstance(self.cumulative_height,np.ndarray) ==  False):
      self.indices = np.zeros(self.shape, dtype = np.int32)
    else:
      calculate_indices(self.cumulative_height, self.z, self.dz, self.nz, self.origin_z, self.indices, self.labelmatrix )


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

  # main_drainage_divide_migration_index = xs.on_demand(description = "Represents the main drainage divide variations at each time step")

  def initialize(self):
    # getting boundary conditions right for the iteratool
    self.node_type = np.ones((self.ny,self.nx), dtype = np.int8)
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
    self.iteratool = iterasator(self.node_type,self.nx,self.ny, self.dx, self.dy)


  def run_step(self):
    # Just checking if self.is_multiple_flow
    self.is_multiple_flow = True if self.receivers.ndim > 1 else False
    if(self.is_multiple_flow == False):
      self.internal_chi = fhl.chiculation_SF(self.stack, self.receivers, self.nb_donors, self.donors, self.lengths, 
        self.elevation.ravel(), self.flowacc.ravel(), self.A_0_chi, self.theta_chi, self.minAcc).reshape(self.ny,self.nx)
    else:
      self.internal_chi = fhl.chiculation_MF(self.stack, self.receivers, self.nb_receivers, self.lengths, self.weights, 
        self.elevation, self.flowacc.ravel(), self.A_0_chi, self.theta_chi, self.minAcc)



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





  # @main_drainage_divide_migration_index.compute
  # def _mainmain_drainage_divide_migration_index():




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
    'label': Label3DFull
})


sediment_model_label3D_SPL = fastscape.models.basic_model.update_processes({
    'bedrock': Bedrock,
    'active_layer': UniformSedimentLayer,
    'init_bedrock': BareRockSurface,
    'flow': MultipleFlowRouter,
    'spl': StreamPowerChannelForeign,
    'label': Label3DSPL
}).drop_processes({'diffusion'})

















































# End of file