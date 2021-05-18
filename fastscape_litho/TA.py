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
####################### Topographic Analysis #################
##############################################################
##############################################################



@xs.process
class QuickTA:
  """
     This Process class provides quick runtime topographic analysis. It analyses the landscapes As if it was a "real" DEM.
     Authors: B.G
  """
    
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
  erosion_rate = xs.foreign(TotalErosion, "rate")

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

  ksnSA_cross_theta = xs.on_demand(
    dims=('n_theta'),
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

  @ksnSA_cross_theta.compute
  def _ksnSA_cross_theta(self):

    dfs = []
    
    thetas = np.arange(0.05,1,0.05)
    dfs = []

    #Getting the basin in Single Flow direction
    basins = fhl.basination_SF(self.fs_context["stack"].astype('int') - 1, self.fs_context["rec"].astype('int') - 1).ravel()
    basin_size = []

    basin_median_X = []
    basin_1stQ_X = []
    basin_3rdQ_X = []

    basin_median_Y = []
    basin_1stQ_Y = []
    basin_3rdQ_Y = []

    basin_median_ksn = []
    basin_1stQ_ksn = []
    basin_3rdQ_ksn = []

    basin_median_DA = []
    basin_1stQ_DA = []
    basin_3rdQ_DA = []

    basin_median_E = []
    basin_1stQ_E = []
    basin_3rdQ_E = []

    nodeids = np.arange(self.nx * self.ny)

    ksnftheta = {}
    for theta in thetas:
      tksn = np.power(self.flowacc.ravel(),theta) * self.get_slope()
      ksnftheta[theta] = tksn

    nbas = 0
    for bs in np.unique(basins):

      mask = (basins == bs) & (self.flowacc.ravel() > self.minAcc)
      if(mask[mask==True].shape[0] == 0):
        continue
      nbas+=1
      basin_size.append(self.flowacc.ravel()[mask].max())
      row,col = self.iteratool.node2rowcol(nodeids[mask])
      X = col * self.nx + self.nx/2
      Y = row * self.ny + self.ny/2
      basin_median_X.append(np.median(X))
      basin_1stQ_X.append(np.percentile(X,25))
      basin_3rdQ_X.append(np.percentile(X,75))
      basin_median_Y.append(np.median(Y))
      basin_1stQ_Y.append(np.percentile(Y,25))
      basin_3rdQ_Y.append(np.percentile(Y,75))
      basin_median_DA.append(np.median(self.flowacc.ravel()[mask]))
      basin_1stQ_DA.append(np.percentile(self.flowacc.ravel()[mask], 25))
      basin_3rdQ_DA.append(np.percentile(self.flowacc.ravel()[mask], 75))
      basin_median_E.append(np.median(self.erosion_rate.ravel()[mask]))
      basin_1stQ_E.append(np.percentile(self.erosion_rate.ravel()[mask], 25))
      basin_3rdQ_E.append(np.percentile(self.erosion_rate.ravel()[mask], 75))

      tbasin_median_ksn = []
      tbasin_1stQ_ksn = []
      tbasin_3rdQ_ksn = []
      for theta in thetas:
        tksn = ksnftheta[theta]
        tbasin_median_ksn.append(np.median(tksn[mask]))
        tbasin_1stQ_ksn.append(np.percentile(tksn[mask], 25))
        tbasin_3rdQ_ksn.append(np.percentile(tksn[mask], 75))

      basin_median_ksn.append(tbasin_median_ksn)
      basin_1stQ_ksn.append(tbasin_1stQ_ksn)
      basin_3rdQ_ksn.append(tbasin_3rdQ_ksn)
      
      

      # output = fhl.chi_contrast_across_divides(main_divide, this_chi.ravel(), self.iteratool, self.main_divide_contrast_distance )
      # dfs.append(pd.DataFrame({"X": output[0], "Y": output[1], "chi_med_zone": output[2], "chi_med_others": output[3],
      #   "chi_max_zone": output[4], "chi_max_others": output[5], "zone": output[6].astype(np.int)}))

    # tds = xr.concat([df.to_xarray() for df in dfs], dim="thetas")
    # tds["thetas"] = thetas
    # nbas = np.unique(basins).ravel().basishape[0]
    tds = xr.Dataset({
      'n_basins': np.arange(nbas),
      'thetas': thetas,
      'basin_size': xr.DataArray(data = basin_size, dims = ('n_basins')),
      'basin_median_X': xr.DataArray(data = basin_median_X, dims = ('n_basins')),
      'basin_1stQ_X': xr.DataArray(data = basin_1stQ_X, dims = ('n_basins')),
      'basin_3rdQ_X': xr.DataArray(data = basin_3rdQ_X, dims = ('n_basins')),
      'basin_median_Y': xr.DataArray(data = basin_median_Y, dims = ('n_basins')),
      'basin_1stQ_Y': xr.DataArray(data = basin_1stQ_Y, dims = ('n_basins')),
      'basin_3rdQ_Y': xr.DataArray(data = basin_3rdQ_Y, dims = ('n_basins')),
      'basin_median_ksn': xr.DataArray(data = basin_median_ksn, dims = ('n_basins', 'thetas')),
      'basin_1stQ_ksn': xr.DataArray(data = basin_1stQ_ksn, dims = ('n_basins', 'thetas')),
      'basin_3rdQ_ksn': xr.DataArray(data = basin_3rdQ_ksn, dims = ('n_basins', 'thetas')),
      'basin_median_DA': xr.DataArray(data = basin_median_DA, dims = ('n_basins')),
      'basin_1stQ_DA': xr.DataArray(data = basin_1stQ_DA, dims = ('n_basins')),
      'basin_3rdQ_DA': xr.DataArray(data = basin_3rdQ_DA, dims = ('n_basins')),
      'basin_median_E': xr.DataArray(data = basin_median_E, dims = ('n_basins')),
      'basin_1stQ_E': xr.DataArray(data = basin_1stQ_E, dims = ('n_basins')),
      'basin_3rdQ_E': xr.DataArray(data = basin_3rdQ_E, dims = ('n_basins'))
    })

    tstep = str(self.step)
    while(len(tstep)<6):
      tstep = "0" +  tstep

    tds.to_zarr(self.output_prefix + "%s_ksn_per_basin_info.zarr"%(tstep))

    return thetas

