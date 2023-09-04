


from deployment.Stabilizer2 import Stabilizer2


stab2 = Stabilizer2()
im = "blender_generated/Ford Mustang 2023 Black-0001.png"
stab2.run(im ,"deployment-outputs/blender_stab/disjoint_version/stab1/"+str( 1)+".json" , "deployment-outputs/blender_stab/disjoint_version/stab2_afterPadding/"+str( 1)+".png")