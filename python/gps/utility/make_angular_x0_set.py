import numpy as np
import random 


# angle range: -pi to pi
# speed range: -1 to 1 by default

def MakeAngularX0Set(n_samp,n_dof,maxvel=1):
   random.seed(1)
   x0 = []
   for i_samp in range(n_samp):
      angs = []
      vels = []
      for i_dof in range(n_dof):
         ra = random.random()
         rv = random.random()
         angs.append( ( ra * 2.0 - 1.0 ) * np.pi )
         vels.append( ( rv * 2.0 - 1.0 ) * maxvel )
      x0.append(np.array(angs + vels))
   return x0

