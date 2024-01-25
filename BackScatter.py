def findBackScatterCoefficients(img, display = False) :
    import numpy as np
    import matplotlib.pyplot as plt
    
    import pywt
    import pywt.data
    from skimage import io, color
    
    # Load image
    data = io.imread(img)
    gray = color.rgb2gray(data)
    coeffs = pywt.dwt2(gray, 'haar')
    
    HH, (HV, VH, VV) = coeffs
    if(display) :
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([HH, HV, VH, VV]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.tight_layout()
        plt.show()
    
    return (HH, HV, VH, VV)

  
def PSNR(original, compressed):
    from math import log10, sqrt
    import numpy as np
    
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        return 0
    max_pixel = original.max()
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return (20-psnr)