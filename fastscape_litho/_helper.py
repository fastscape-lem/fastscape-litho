import numpy as np
import numba as nb
from ._looper import iterasator


@nb.njit
def _flow_accumulate_sd(field, stack, receivers):
  for inode in stack[-1::-1]:
    if receivers[inode] != inode:
      field[receivers[inode]] += field[inode]
    if(field[inode] < 0):
      field[inode] = 0


@nb.njit
def _flow_accumulate_mfd(field, stack, nb_receivers, receivers, weights):
  for inode in stack:
    if nb_receivers[inode] == 1 and receivers[inode, 0] == inode:
      continue

    for k in range(nb_receivers[inode]):
      irec = receivers[inode, k]
      field[irec] += field[inode] * weights[inode, k]
    if(field[inode] < 0):
      field[inode] = 0

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

    # chi[inode] = chi[receivers[inode]] + lengths[inode]/2 * ( ((A0/area[inode]) ** theta)  + ((A0/area[receivers[inode]]) ** theta) ) 
    chi[inode] = chi[receivers[inode]] + lengths[inode] * ( ((A0/area[inode]) ** theta)) 

  return chi


@nb.njit()
def chiculation_MF(mstack, receivers, nb_receivers, lengths, weights, elevation, area, A0, theta, minAcc):


  chi = np.zeros_like(mstack, dtype = np.float64)
  for inode in mstack[::-1]:

    if(nb_receivers[inode] == 0 or area[inode] < minAcc):
      continue

    chi[inode] = 0
    for j in range(nb_receivers[inode]):
      # chi[inode] += weights[inode,j] * (chi[receivers[inode,j]] + lengths[inode,j]/2 * ( ((A0/area[inode]) ** theta)  + ((A0/area[receivers[inode,j]]) ** theta) ) )
      chi[inode] += weights[inode,j] * (chi[receivers[inode,j]] + lengths[inode,j] * ( ((A0/area[inode]) ** theta)) )

  return chi


@nb.njit()
def ksn_calculation_SF(elevation, chi, receivers, stack):
  
  ksn = np.zeros_like(chi.ravel())

  for i in range(chi.shape[0]):
    irec = receivers[i]
    if(irec == i or chi[i] == 0):
      # n_ignored += 1
      continue

    ksn[i] = elevation[i] - elevation[irec]
    ksn[i] = ksn[i]/(chi[i] - chi[irec])
    # if(ksn[i] == 0):
    #   n_0 += 1

    if(ksn[i]<0):
      ksn[i] = 0
  # print(n_ignored)

  return ksn

@nb.njit()
def ksn_calculation_MF(elevation, chi, receivers,nb_receivers, weights, stack):
  
  ksn = np.zeros_like(chi.ravel())

  for i in range(chi.shape[0]):
    
    if(chi[i] == 0):
        continue

    this_ksn = 0

    for j in range(nb_receivers[i]):
      irec = receivers[i,j]

      this_this_ksn = elevation[i] - elevation[irec]
      this_this_ksn = this_this_ksn/(chi[i] - chi[irec])
      this_ksn += weights[i,j] * this_this_ksn

      # if(ksn[i] == 0):
      #   n_0 += 1
    # print(n_ignored)
    ksn[i] = this_ksn
    if(ksn[i]<0):
      ksn[i] = 0

  return ksn


@nb.njit()
def slope_SF(elevation, receivers, lengths):
  slope = np.zeros_like(elevation)
  for i in range(receivers.shape[0]):
    if(i == receivers[i]):
      continue
    slope[i] = (elevation[i] - elevation[receivers[i]])/lengths[i]
  return slope

@nb.njit()
def slope_MF(elevation, receivers, lengths, nb_receivers, weights):
  slope = np.zeros_like(elevation)
  for i in range(receivers.shape[0]):
    if(nb_receivers[i] == 0):
      continue

    for j in range(nb_receivers[i]):
      slope[i] += weights[i,j] *  ((elevation[i] - elevation[receivers[i,j]])/lengths[i,j])
    # slope[i] = slope[i]/nb_receivers[i]
  return slope


@nb.njit()
def basination_SF(Sstack, receivers):
  basins = np.zeros_like(Sstack, dtype = np.int32)
  basin_ID = -1
  for inode in Sstack:
    if(inode == receivers[inode]):
      basin_ID = inode
    basins[inode] = basin_ID
  return basins

@nb.njit()
def is_draiange_divide_SF(basins, iterator):
  is_DD = np.zeros_like(basins, dtype = nb.boolean)
  for i in range(basins.shape[0]):
    val = basins[i]
    for j in iterator.get_neighbouring_nodes(i):
      if(j < 0 or is_DD[i] == 1):
        continue
      if(basins[j] != val):
        is_DD[i] = 1
        is_DD[j] = 1
  return is_DD


@nb.njit()
def extract_main_regions(stack, main_divide, receivers):

  # First step: labelling the thingy
  for i in stack:
    main_divide[i] = main_divide[receivers[i]]


@nb.njit()
def boundcheck(node, iterator):
  r, c = iterator.node2rowcol(node)
  if(r < 0 or r >= iterator.ny or c < 0 or c >= iterator.nx):
    return False
  else:
    return True



@nb.njit()
def chi_contrast_across_divides(main_divide, chi, iterator, distance ):
  
  # Formatting the output by index:
  # 0 -> X
  # 1 -> Y
  # 2 -> chi_median_zone
  # 3 -> chi_median_other
  # 4 -> chi_max_zone
  # 5 -> chi_max_other
  # 6 -> zone
  output = np.zeros((7,iterator.ny *iterator.nx ), dtype = nb.float64)
  # Number of drainage divide points
  n_elements = 0
  # The half window in integer size 
  tdx_d2 = round(distance/iterator.dx/2)
  tdy_d2 = round(distance/iterator.dy/2)

  # Iterating through all nodes
  for i in range(main_divide.shape[0]):
    # Value of the current zone
    val = main_divide[i]
    # Gathering chi values of the different zones
    vals_zone = []
    vals_other = []

    # getting the row and col index
    r, c = iterator.node2rowcol(i) 
    # Checker: if all the points gathered are within the same zone, I ignore the point
    keep = False

    # weeeeeeeee itarating aroud the node
    for tr in np.arange(r - tdy_d2,r + tdy_d2 + tdy_d2/2, 1, dtype = nb.int32):
      for tc in  np.arange(c - tdx_d2,c + tdx_d2 + tdx_d2/2, 1, dtype = nb.int32):
        # falling back into node index
        j = iterator.nx * tr + tc 

        #Chacking whether the node exists
        if(boundcheck(j,iterator) == False):
          continue

        #If the node is in a different zone: I gather the information
        if(main_divide[j] != val):
          keep = True
          # As well as saving chi for this area
          if(chi[j] > 0):
            vals_other.append(chi[j])
        else:
          # else just saving chi for the otehr area
          if(chi[j] > 0):
            vals_zone.append(chi[j])

    #if at least one node is in the other area and not null
    if(keep and len(vals_other)>0 and len(vals_zone) > 0):
      arrz = np.zeros(len(vals_zone))
      arro = np.zeros(len(vals_other))
      for I in range(len(vals_zone)):
        arrz[I] = vals_zone[I]
      for I in range(len(vals_other)):
        arro[I] = vals_other[I]

      # calculating the output as described above
      output[0,n_elements] = c * iterator.dx + iterator.dx/2
      output[1,n_elements] = r * iterator.dy + iterator.dy/2
      output[2,n_elements] = np.median(arrz)
      output[3,n_elements] = np.median(arro)
      output[4,n_elements] = np.max(arrz)
      output[5,n_elements] = np.max(arro)
      output[6,n_elements] = nb.float64(main_divide[i])
      n_elements += 1
  # Done, only returning the outputs needed
  return output[:,:n_elements]































  # End of file