import numpy as np
import numba as nb


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

    chi[inode] = chi[receivers[inode]] + lengths[inode]/2 * ( ((A0/area[inode]) ** theta)  + ((A0/area[receivers[inode]]) ** theta) ) 

  return chi


@nb.njit()
def chiculation_MF(mstack, receivers, nb_receivers, lengths, weights, elevation, area, A0, theta, minAcc):


  chi = np.zeros_like(mstack, dtype = np.float64)
  for inode in mstack[::-1]:

    if(nb_receivers[inode] == 0 or area[inode] < minAcc):
      continue

    chi[inode] = 0
    for j in range(nb_receivers[inode]):
      chi[inode] += weights[inode,j] * (chi[receivers[inode,j]] + lengths[inode,j]/2 * ( ((A0/area[inode]) ** theta)  + ((A0/area[receivers[inode,j]]) ** theta) ) )

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