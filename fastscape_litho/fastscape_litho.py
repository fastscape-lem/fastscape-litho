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




@nb.njit()
def chiculation_SF(stack, receivers, nb_donors, donors, lengths, elevation, area, A0, theta, minAcc):

  # First I am getting the length to donors
  # length2donors = np.full_like(donors,-1)
  # for inode in stack:
  #   for idon in range(nb_donors[inode]):
  #     length2donors[inode,idon] = lengths[donors[inode,idon]]

  chi = np.zeros_like(stack, dtype = np.float64)
  for inode in stack:
    if(inode == receivers[inode] or area[inode] < minAcc):
      continue
    chi[inode] = chi[receivers[inode]] + lengths[inode]/2 * ((A0/area[inode]) ** theta  - (A0/area[receivers[inode]]) ** theta ) 
  return chi

@nb.njit()
def ksn_calculation_SF(elevation, chi, receivers):
  ksn = np.zeros_like(chi)
  for i in range(chi.shape[0]):
    irec = receivers [i]
    if(irec == i or chi[i] == 0):
      continue
      
    ksn[i] = elevation[i] - elevation[irec]
    ksn[i] = ksn[i]/(chi[i] - chi[irec])

    if(ksn[i]<0):
      ksn[i] = 0
  return ksn



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

  # FlowRouter
  elevation = xs.foreign(SurfaceToErode, 'elevation')
  stack = xs.foreign(FlowRouter, 'stack')
  nb_receivers = xs.foreign(FlowRouter, 'nb_receivers')
  receivers = xs.foreign(FlowRouter, 'receivers')
  lengths = xs.foreign(FlowRouter, 'lengths')
  weights = xs.foreign(FlowRouter, 'weights')
  nb_donors = xs.foreign(FlowRouter, 'nb_donors')
  donors = xs.foreign(FlowRouter, 'donors')

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
    description='Local k_sn index'
    )

  def run_step(self):
    # Just checking if self.is_multiple_flow
    self.is_multiple_flow = True if self.receivers.ndim > 1 else False
    if(self.is_multiple_flow == False):
      self.internal_chi = chiculation_SF(self.stack, self.receivers, self.nb_donors, self.donors, self.lengths, 
        self.elevation.ravel(), self.flowacc.ravel(), self.theta_chi, self.A_0_chi, self.minAcc).reshape(self.ny,self.nx)


  @chi.compute
  def _chi(self):
    if(self.is_multiple_flow == False):
      return self.internal_chi.reshape(self.ny,self.nx)
    else:
      return np.zeros_like(self.elevation)
    

  @ksn.compute
  def _ksn(self):
    if(self.is_multiple_flow == False):
      return ksn_calculation_SF(self.elevation.ravel(), self.internal_chi, self.receivers).reshape(self.ny,self.nx)
    else:
      return np.zeros_like(self.elevation)




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