import ee
import BackScatter as scatter
import matplotlib.pyplot as plt

# Trigger the authentication flow.
#ee.Authenticate()

# Initialize the library.
ee.Initialize()

i_dates = ['2022-07-01']
f_dates = ['2022-08-01']

u_lon = 78.6022
u_lat = 20.7453

vv_snr = []
vh_snr = []
mode_val = 'VV'

for count in range(0, len(i_dates)) :
    # Import the MODIS land cover collection.
    lc = ee.ImageCollection('MODIS/006/MCD12Q1')
    
    # Import the MODIS land surface temperature collection.
    lst = ee.ImageCollection('COPERNICUS/S1_GRD')
    #lst = ee.ImageCollection('COPERNICUS/S1_GRD')
    
    # Import the USGS ground elevation image.
    elv = ee.Image('USGS/SRTMGL1_003')

    # Initial date of interest (inclusive).
    i_date = i_dates[count]
    
    # Final date of interest (exclusive).
    f_date = f_dates[count]
    
    # Selection of appropriate bands and dates for LST.
    lst = lst.filter(ee.Filter.listContains('transmitterReceiverPolarisation', mode_val)).filter(ee.Filter.eq('instrumentMode', 'IW')).select(mode_val).filterDate(i_date, f_date)
    
    # Define the urban location of interest as a point near given area
    u_poi = ee.Geometry.Point(u_lon, u_lat)
    
    # Define the rural location of interest as a point away from the city.
    r_lon = u_lon
    r_lat = u_lat
    r_poi = ee.Geometry.Point(r_lon, r_lat)
    
    scale = 1000  # scale in meters
    
    # Print the elevation
    elv_urban_point = elv.sample(u_poi, scale).first().get('elevation').getInfo()
    print('Ground elevation at urban point:', elv_urban_point, 'm')
    
    # Calculate and print the mean value of the LST collection at the point.
    lst_urban_point = lst.mean().sample(u_poi, scale).first().get(mode_val).getInfo()
    print('Average daytime LST at urban point:', round(lst_urban_point*0.02 -273.15, 2), '°C')
    
    # Print the land cover type at the point.
    lc_urban_point = lc.first().sample(u_poi, scale).first().get('LC_Type1').getInfo()
    print('Land cover value at urban point is:', lc_urban_point)
    
    # Get the data for the pixel intersecting the point in urban area.
    #lst_u_poi = lst.getRegion(u_poi, scale).getInfo()
    
    # Get the data for the pixel intersecting the point in rural area.
    #lst_r_poi = lst.getRegion(r_poi, scale).getInfo()
    
    # Preview the result.
    #lst_u_poi[:5]
    
    import pandas as pd
    
    def ee_array_to_df(arr, list_of_bands):
        """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
        df = pd.DataFrame(arr)
    
        # Rearrange the header.
        headers = df.iloc[0]
        df = pd.DataFrame(df.values[1:], columns=headers)
    
        # Remove rows without data inside.
        df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()
    
        # Convert the data to numeric values.
        for band in list_of_bands:
            df[band] = pd.to_numeric(df[band], errors='coerce')

        # Convert the time field into a datetime.
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    
        # Keep the columns of interest.
        df = df[['time','datetime',  *list_of_bands]]
    
        return df
    
    #lst_df_urban = ee_array_to_df(lst_u_poi,[mode_val])
    
    def t_modis_to_celsius(t_modis):
        """Converts MODIS LST units to degrees Celsius."""
        t_celsius =  0.02*t_modis - 273.15
        return t_celsius
    
    # Apply the function to get temperature in celsius.
    #lst_df_urban[mode_val] = lst_df_urban[mode_val].apply(t_modis_to_celsius)
    
    # Do the same for the rural point.
    #lst_df_rural = ee_array_to_df(lst_r_poi,[mode_val])
    #lst_df_rural[mode_val] = lst_df_rural[mode_val].apply(t_modis_to_celsius)
    
    #lst_df_urban.head()
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import optimize
    #%matplotlib inline
    
    # Fitting curves.
    ## First, extract x values (times) from the dfs.
    #x_data_u = np.asanyarray(lst_df_urban['time'].apply(float))  # urban
    #x_data_r = np.asanyarray(lst_df_rural['time'].apply(float))  # rural
    
    ## Secondly, extract y values (LST) from the dfs.
    #y_data_u = np.asanyarray(lst_df_urban[mode_val].apply(float))  # urban
    #y_data_r = np.asanyarray(lst_df_rural[mode_val].apply(float))  # rural
    
    ## Then, define the fitting function with parameters.
    def fit_func(t, lst0, delta_lst, tau, phi):
        return lst0 + (delta_lst/2)*np.sin(2*np.pi*t/tau + phi)
    
    
    roi = u_poi.buffer(1e6)
    
    # Reduce the LST collection by mean.
    lst_img = lst.mean()
    
    # Adjust for scale factor.
    lst_img = lst_img.select(mode_val).multiply(0.02)
    
    # Convert Kelvin to Celsius.
    lst_img = lst_img.select(mode_val).add(-273.15)
    
    from IPython.display import Image
    
    
    lyon = u_poi.buffer(10000)  # meters
    
    #url = elv_img.getThumbUrl({
    #    'min': 150, 'max': 350, 'region': lyon, 'dimensions': 512,
    #    'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']})
    #Image(url=url)
    
    link = lst_img.getDownloadURL({
        'scale': 20+count,
        'crs': 'EPSG:4326',
        'fileFormat': 'GeoTIFF',
        'region': lyon})
    print(link)
    
    print('Downloading...')
    zip_name = "downloaded.zip"
    out_folder = 'extracted'
    out_file = out_folder + "/"+ 'download.' + mode_val  + '.tif';
    proc_file = 'output.tif'
    
    import urllib.request
    urllib.request.urlretrieve(link, zip_name)
    
    print('Extracting...')
    import zipfile
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(out_folder)
        
    print('Files unzipped to ' + out_folder + '/ folder')
    
    from skimage import io
    import matplotlib.pyplot as plt
    
    # read the image stack
    img = io.imread(out_file)
    
    img2 = (img-np.amin(img))/(np.amax(img)-np.amin(img))
    img2 = np.around(img2*255)
    img2 = img2.astype(np.uint8)
    # show the image
    plt.imshow(img2)
    plt.axis('off')
    # save the image
    plt.savefig(proc_file, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.0)
    (hh, vh, vv, hv) = scatter.findBackScatterCoefficients(proc_file, True)
    
    snr_vh = scatter.PSNR(hh, vh)
    snr_vv = scatter.PSNR(hh, vv)

    vv_snr.append(snr_vv)
    vh_snr.append(snr_vh)
    
    frame_bkp = img2
    rows = len(frame_bkp)
    cols = len(frame_bkp[0])
    
    outFrame = np.empty((rows, cols, 3))
    
    strVal = 'Red: Land, Green: Forest, Blue: Water, RG:Barren, Green Blue: Urban'
    areaLand = 0
    areaForest = 0
    areaWater = 0
    areaBarren = 0
    areaUrban = 0
    
    for row in range(0, rows) :
        for col in range(0, cols) :
            intensity = frame_bkp[row, col]
            
            if(intensity < 25) :
                #Low intensity land regions
                red = 255
                green = 0
                blue = 0
                areaLand = areaLand + 1
                
            elif(intensity < 50) :
                #Moderate intensity regions
                red = 0
                green = 255
                blue = 0
                
                areaForest = areaForest + 1
            elif(intensity < 80) :
                #High intensity water regions
                red = 0
                green = 0
                blue = 255
                
                areaWater = areaWater + 1
            elif(intensity < 120) :
                #Barren
                red = 0
                green = 255
                blue = 255
                
                areaBarren = areaBarren + 1
            elif(intensity < 150) :
                #Barren
                red = 255
                green = 255
                blue = 0
                
                areaBarren = areaBarren + 1
            else :
                #Urban
                red = 255
                green = 0
                blue = 255
                
                areaUrban = areaUrban + 1
                
            outFrame[row, col, 0] = red
            outFrame[row, col, 1] = green
            outFrame[row, col, 2] = blue

    areaLand = areaLand * 100 / (rows*cols)
    areaForest = areaForest * 100 / (rows*cols)
    areaWater = areaWater * 100 / (rows*cols)
    areaBarren = areaBarren * 100 / (rows*cols)
    areaUrban = areaUrban * 100 / (rows*cols)
    plt.imshow(outFrame)
    plt.title(strVal)
    
    print('Land Area %0.04f acres' % areaLand)
    print('Forest Area %0.04f acres' % areaForest)
    print('Water Area %0.04f acres' % areaWater)
    print('Barren Area %0.04f acres' % areaBarren)
    print('Urban Area %0.04f acres' % areaUrban)
    
    total = areaLand + areaForest + areaWater + areaBarren + areaUrban
    waterPer = areaWater / total
    landPer = 1 - waterPer
    
    if(waterPer > landPer) :
        print('Higher Water Quality Index')
    else :
        print('Higher Land Quality Index')
#plt.plot(vv_snr)
#plt.show()
#plt.figure()
#plt.plot(vh_snr)
#plt.show()
