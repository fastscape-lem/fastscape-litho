"""
This module contains the xarray-simlab processes related to the 3D cube of lithology.
B.G.
"""
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
####################### LABEL 3D #############################
##############################################################
##############################################################

@nb.njit()
def calculate_indices(cumulative_height, z, dz, nz, oz, indices, labelmatrix ):
  '''
    Helper function calculating the intercept indices from the 3D matrix functio of the depth
  '''
  # Iterating through the rows (i) and col (j)
  for i in range(indices.shape[0]):
    for j in range(indices.shape[1]):

      this_id = math.ceil((cumulative_height[i,j] - oz)/dz)
      if(this_id >= nz):
        indices[i,j] = -1
      else:
        indices[i,j] = labelmatrix[i,j,this_id]




@xs.process
class Label3D:
  '''
    Base class for the 3D label, it only manages the geometry of it: ingesting a 3D matrix of indices,
    and managing the intersection with the topography and erosion.
    Processes with variable characteristics need to be registered separatedly.
    Created: 03/2021
    Last Mod.: 03/2021
    Authors: B.G.  
  '''

  # First defining the variable belonging to Label3D
  dz = xs.variable(
    default = 50, 
    description = 'Depth resolution of the 3D matrix of labels'
  )

  origin_z = xs.variable(
    default = 0,
    description = 'original elevation of the grid. Leave to 0'
  )

  z = xs.index(
    dims='z',
    description='Matrix depth coordinate'
  )

  labelmatrix = xs.variable(
    dims = ('y','x','z'),
    description = "The 3D numpy array of indices. dimensions are row, col, depth indice."
  )

  indices = xs.variable(
    intent = 'out',
    dims=[(), ('y', 'x')],
    description= 'Indices of the depth matrix intersection at runtime, it is used to get the 2D arrays of rock-dependent characteristics.'
  )


  # Some foreign variables needed for the computing
  ## Cumulative height is the cumulative erosion into bedrock 
  cumulative_height = xs.foreign(TotalErosion, "cumulative_height")
  ## Shape is the shape of the 2D array of topography
  shape = xs.foreign(UniformRectilinearGrid2D,"shape")


  def initialize(self):
    # initialising the depth geometry
    self.nz = self.labelmatrix.shape[2]
    self.z = np.linspace(-1 * self.origin_z, -1 * self.origin_z + self.nz * self.dz + self.dz , self.nz)
    # Adn the indices
    self.indices = np.full(self.shape, -1, dtype = np.int32)


  def run_step(self):
    #At the start, there is no array so the original depth is 0
    if(isinstance(self.cumulative_height,np.ndarray) ==  False):
      self.indices = np.zeros(self.shape, dtype = np.int32)
    else:
      # Else I am calculating the intercept calling the numba function above
      calculate_indices(self.cumulative_height, self.z, self.dz, self.nz, self.origin_z, self.indices, self.labelmatrix )



##############################################################
##############################################################
########### Adaptation of fastscape processes ################
##############################################################
##############################################################


@xs.process
class DifferentialStreamPowerChannelTDForeign(DifferentialStreamPowerChannelTD):
  k_coef_bedrock = xs.variable(intent = 'out',
    dims=[(), ('y', 'x')],
    description='bedrock channel incision coefficient'
  )
  Kr_lab = xs.variable(dims = 'n_labels', description = 'Kr value for each label')
  indices = xs.foreign(Label3D, "indices")

  def run_step(self):
    self.k_coef_bedrock = self.Kr_lab[self.indices]
    super(DifferentialStreamPowerChannelTDForeign, self).run_step()


@xs.process
class DifferentialLinearDiffusionForeign(DifferentialLinearDiffusion):
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

  Kdr_lab = xs.variable(dims = 'n_labels', description = 'Kdr value for each label')
  Kds_lab = xs.variable(dims = 'n_labels', description = 'Kds value for each label')

  indices = xs.foreign(Label3D, "indices")

  def run_step(self):
    self.diffusivity_bedrock = self.Kdr_lab[self.indices]
    self.diffusivity_soil = self.Kds_lab[self.indices]
    super(DifferentialLinearDiffusionForeign, self).run_step()

@xs.process
class StreamPowerChannelForeign(StreamPowerChannel):
  k_lab = xs.variable(dims = 'n_labels', description = 'K value for each label')
  k_coef = xs.variable(intent = 'out',
    dims=[(), ('y', 'x')],
    description='bedrock channel incision coefficient'
  )
  indices = xs.foreign(Label3D, "indices")

  def run_step(self):
    self.k_coef = self.k_lab[self.indices]
    super(StreamPowerChannelForeign, self).run_step()

@xs.process
class FlowAccumulatorForeign(FlowAccumulator):
  """Accumulate the flow from upstream to downstream."""

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

  shape = xs.foreign(UniformRectilinearGrid2D, 'shape')
  cell_area = xs.foreign(UniformRectilinearGrid2D, 'cell_area')
  stack = xs.foreign(FlowRouter, 'stack')
  nb_receivers = xs.foreign(FlowRouter, 'nb_receivers')
  receivers = xs.foreign(FlowRouter, 'receivers')
  weights = xs.foreign(FlowRouter, 'weights')
  indices = xs.foreign(Label3D, "indices")


  flowacc = xs.variable(
      dims=('y', 'x'),
      intent='out',
      description='flow accumulation from up to downstream'
  )

  def run_step(self):
    self.runoff = self.precipitations - self.infiltration_lab[self.indices]

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


  # UniformRectilinearGrid2D

##############################################################
##############################################################
####################### Models ###############################
##############################################################
##############################################################



# A model with 3D litho block and diffusion/sediment stream power
sediment_model_label3D = fastscape.models.basic_model.update_processes({
    'bedrock': Bedrock,
    'active_layer': UniformSedimentLayer,
    'init_bedrock': BareRockSurface,
    'flow': MultipleFlowRouter,
    'spl': DifferentialStreamPowerChannelTDForeign,
    'diffusion': DifferentialLinearDiffusionForeign,
    'label': Label3D,
    'drainage': DrainageArea
})

#A fast, SPL single flow only model for quick tests 
fast_SPL_model_label3D = fastscape.models.basic_model.update_processes({
    'bedrock': Bedrock,
    'active_layer': UniformSedimentLayer,
    'init_bedrock': BareRockSurface,
    'flow': SingleFlowRouter,
    'spl': StreamPowerChannelForeign,
    'label': Label3D,
    'drainage': DrainageArea
}).drop_processes({'diffusion'})

# A basic set up for 3D block without any surface process activated (can be used to create new models)
basic_model_label3D = fast_SPL_model_label3D.drop_processes(['spl'])

# A model utilising the full power of the 3D block possibilities
full_model_label3D = fastscape.models.basic_model.update_processes({
    'bedrock': Bedrock,
    'active_layer': UniformSedimentLayer,
    'init_bedrock': BareRockSurface,
    'flow': MultipleFlowRouter,
    'spl': DifferentialStreamPowerChannelTDForeign,
    'diffusion': DifferentialLinearDiffusionForeign,
    'label': Label3D,
    'drainage': DrainageArea
})














































# End of file