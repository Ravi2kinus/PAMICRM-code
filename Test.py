import ee
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def load_dataset(ImageCollection_ID,begin,end,aoi):
    ic = ee.ImageCollection(ImageCollection_ID).filterDate(begin,end).filterBounds(aoi)
    return ic

def filter_sentinel1(ImageCollection,polarisation,instrumentMode,resolution):
    ic = ImageCollection.filter(ee.Filter.listContains('transmitterReceiverPolarisation',polarisation)).filter(ee.Filter.eq('instrumentMode',instrumentMode)).filterMetadata('resolution_meters','equals', resolution)
    return ic

def seperate_look_angels(ImageCollection,polarisation):
    Ascending = ImageCollection.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).select(polarisation)
    Descending = ImageCollection.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).select(polarisation)
    return Ascending,Descending


def get_properties(ImageCollection):
    features = ImageCollection.getInfo()['features']
    dict_list = []
    for f in features:
        prop = f['properties']
        dict_list.append(prop)
    df = pd.DataFrame.from_records(dict_list).drop(['system:footprint','transmitterReceiverPolarisation'],axis=1)
    #Pandas Series of unique distinc values in df
    unique = df.nunique()
    im_id_list = [item.get('id') for item in ImageCollection.getInfo().get('features')]
    date_list = [datetime.datetime.strptime(x[35:43],'%Y%m%d') for x in im_id_list]
    #property_names = list(df.columns.values) 
    return unique, im_id_list, date_list

def make_mosaic(date):
    date = ee.Date(date['value'])
    filterCollection = VV_Ascending.filterDate(date, date.advance(1,'day'))
    image = ee.Image(filterCollection.mosaic()).copyProperties(filterCollection.first(),["system:time_start"])
    return image

#Time of interest
begin = ee.Date.fromYMD(2016,1,1)
end = ee.Date.fromYMD(2016,1,20)
date_range = end.difference(begin, 'day')
u_lon = 77.5686
u_lat = 19.9104
u_poi = ee.Geometry.Point(u_lon, u_lat)

#Source dataset
ried_225_222 = ee.FeatureCollection('users/tillmueller1990/ried_225_222')
sentinel1 = load_dataset('COPERNICUS/S1_GRD',begin,end,ried_225_222)
#Filter dataset for High resolution and Vertical transmitt vertical receive
sentinel1_VV = filter_sentinel1(sentinel1,'VV','IW',10)
#Filter for different look angles
VV_Ascending,VV_Descending = seperate_look_angels(sentinel1_VV,'VV')

#Get list of ids,dates and unique count of prop
unique, im_id_list, date_list = get_properties(VV_Ascending)
date_list = ee.List([ee.Date(x) for x in date_list])
newList = ee.List([])

for date in date_list.getInfo():
    mosaic = ee.Image(make_mosaic(date))
    print(mosaic.getInfo())
    link = mosaic.getDownloadURL({
            'scale': 20,
            'fileFormat': 'GeoTIFF',
            'region': u_poi})
    print(link)
    
    print('Downloading...')
    zip_name = "downloaded.zip"
    out_folder = 'extracted'
    out_file = out_folder + "/"+ 'download.VV.tif';
    proc_file = 'output.tif'
    
    import urllib.request
    urllib.request.urlretrieve(link, zip_name)
    
    print('Extracting...')
    import zipfile
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(out_folder)
    
    print('Files unzipped to ' + out_folder + '/ folder')
    
    from skimage import io
    
    # read the image stack
    img = io.imread(out_file)
    
    fname = 'dataset/' + str(date['value']) + ".png"
    io.imsave(fname, img)
    # show the image
    plt.imshow(img)
    plt.axis('off')