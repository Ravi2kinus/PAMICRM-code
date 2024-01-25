import ee
import BackScatter as scatter
import matplotlib.pyplot as plt

# Trigger the authentication flow.
#ee.Authenticate()

# Initialize the library.
#ee.Initialize()

# Import the MODIS land cover collection.
lc = ee.ImageCollection('MODIS/006/MCD12Q1')

# Import the MODIS land surface temperature collection.
lst = ee.ImageCollection('MODIS/006/MOD11A1')
#lst = ee.ImageCollection('COPERNICUS/S1_GRD')

# Import the USGS ground elevation image.
elv = ee.Image('USGS/SRTMGL1_003')
i_dates = ['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01']
f_dates = ['2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01']

vv_snr = []
vh_snr = []

for count in range(0, len(i_dates)) :
    # Initial date of interest (inclusive).
    i_date = i_dates[count]
    
    # Final date of interest (exclusive).
    f_date = f_dates[count]
    
    # Selection of appropriate bands and dates for LST.
    lst = lst.select('LST_Day_1km', 'QC_Day').filterDate(i_date, f_date)
    
    # Define the urban location of interest as a point near given area
    u_lon = 79.088158
    u_lat = 21.145800
    u_poi = ee.Geometry.Point(u_lon, u_lat)
    
    # Define the rural location of interest as a point away from the city.
    r_lon = 79.188158
    r_lat = 21.145800
    r_poi = ee.Geometry.Point(r_lon, r_lat)
    
    scale = 1000  # scale in meters
    
    # Print the elevation near Lyon, France.
    elv_urban_point = elv.sample(u_poi, scale).first().get('elevation').getInfo()
    print('Ground elevation at urban point:', elv_urban_point, 'm')
    
    # Calculate and print the mean value of the LST collection at the point.
    lst_urban_point = lst.mean().sample(u_poi, scale).first().get('LST_Day_1km').getInfo()
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
    
    #lst_df_urban = ee_array_to_df(lst_u_poi,['LST_Day_1km'])
    
    def t_modis_to_celsius(t_modis):
        """Converts MODIS LST units to degrees Celsius."""
        t_celsius =  0.02*t_modis - 273.15
        return t_celsius
    
    # Apply the function to get temperature in celsius.
    #lst_df_urban['LST_Day_1km'] = lst_df_urban['LST_Day_1km'].apply(t_modis_to_celsius)
    
    # Do the same for the rural point.
    #lst_df_rural = ee_array_to_df(lst_r_poi,['LST_Day_1km'])
    #lst_df_rural['LST_Day_1km'] = lst_df_rural['LST_Day_1km'].apply(t_modis_to_celsius)
    
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
    #y_data_u = np.asanyarray(lst_df_urban['LST_Day_1km'].apply(float))  # urban
    #y_data_r = np.asanyarray(lst_df_rural['LST_Day_1km'].apply(float))  # rural
    
    ## Then, define the fitting function with parameters.
    def fit_func(t, lst0, delta_lst, tau, phi):
        return lst0 + (delta_lst/2)*np.sin(2*np.pi*t/tau + phi)
    
    ## Optimize the parameters using a good start p0.
    #lst0 = 20
    #delta_lst = 40
    #tau = 365*24*3600*1000   # milliseconds in a year
    #phi = 2*np.pi*4*30.5*3600*1000/tau  # offset regarding when we expect LST(t)=LST0
    
    #params_u, params_covariance_u = optimize.curve_fit(
    #    fit_func, x_data_u, y_data_u, p0=[lst0, delta_lst, tau, phi])
    #params_r, params_covariance_r = optimize.curve_fit(
    #    fit_func, x_data_r, y_data_r, p0=[lst0, delta_lst, tau, phi])
    
    # Subplots.
    #fig, ax = plt.subplots(figsize=(14, 6))
    
    # Add scatter plots.
    #ax.scatter(lst_df_urban['datetime'], lst_df_urban['LST_Day_1km'],
    #           c='black', alpha=0.2, label='Urban (data)')
    #ax.scatter(lst_df_rural['datetime'], lst_df_rural['LST_Day_1km'],
    #           c='green', alpha=0.35, label='Rural (data)')
    
    # Add fitting curves.
    #ax.plot(lst_df_urban['datetime'],
    #        fit_func(x_data_u, params_u[0], params_u[1], params_u[2], params_u[3]),
    #        label='Urban (fitted)', color='black', lw=2.5)
    #ax.plot(lst_df_rural['datetime'],
    #        fit_func(x_data_r, params_r[0], params_r[1], params_r[2], params_r[3]),
    #        label='Rural (fitted)', color='green', lw=2.5)
    
    # Add some parameters.
    #ax.set_title('Daytime Land Surface Temperature Near Lyon', fontsize=16)
    #ax.set_xlabel('Date', fontsize=14)
    #ax.set_ylabel('Temperature [C]', fontsize=14)
    #ax.set_ylim(-0, 40)
    #ax.grid(lw=0.2)
    #ax.legend(fontsize=14, loc='lower right')
    
    #plt.show()
    
    roi = u_poi.buffer(1e6)
    
    # Reduce the LST collection by mean.
    lst_img = lst.mean()
    
    # Adjust for scale factor.
    lst_img = lst_img.select('LST_Day_1km').multiply(0.02)
    
    # Convert Kelvin to Celsius.
    lst_img = lst_img.select('LST_Day_1km').add(-273.15)
    
    from IPython.display import Image
    
    # Create a URL to the styled image for a region around France.
    #url = lst_img.getThumbUrl({
    #    'min': 10, 'max': 30, 'dimensions': 512, 'region': roi,
    #    'palette': ['blue', 'yellow', 'orange', 'red']})
    #print(url)
    
    # Display the thumbnail land surface temperature in France.
    #print('\nPlease wait while the thumbnail loads, it may take a moment...')
    #Image(url=url)
    
    # Make pixels with elevation below sea level transparent.
    #elv_img = elv.updateMask(elv.gt(0))
    
    # Display the thumbnail of styled elevation in France.
    #Image(url=elv_img.getThumbURL({
    #    'min': 0, 'max': 2000, 'dimensions': 512, 'region': roi,
    #    'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}))
    
    # Create a buffer zone of 10 km around Lyon.
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
    out_file = out_folder + "/"+ 'download.LST_Day_1km.tif';
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
    # show the image
    plt.imshow(img)
    plt.axis('off')
    # save the image
    plt.savefig(proc_file, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.0)
    (hh, vh, vv, hv) = scatter.findBackScatterCoefficients(proc_file, True)
    
    snr_vh = scatter.PSNR(hh, vh)
    snr_vv = scatter.PSNR(hh, vv)

    vv_snr.append(snr_vv)
    vh_snr.append(snr_vh)

plt.plot(vv_snr)
plt.show()
plt.figure()
plt.plot(vh_snr)
plt.show()
