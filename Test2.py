###################################################################################
# Setup Earth Engine.
###################################################################################

import ee
#ee.Authenticate()
#ee.Initialize()

###################################################################################
# Get the animation URL.
###################################################################################

# Define an area of interest geometry with a global non-polar extent.
aoi = ee.Geometry.Polygon(
  [[[-179.0, 78.0], [-179.0, -58.0], [179.0, -58.0], [179.0, 78.0]]], None,
  False)

# Import hourly predicted temperature image collection for northern winter
# solstice. Note that predictions extend for 384 hours; limit the collection
# to the first 24 hours.
temp_col = (ee.ImageCollection('NOAA/GFS0P25')
  .filterDate('2018-12-22', '2018-12-23')
  .limit(24)
  .select('temperature_2m_above_ground'))

# Define arguments for animation function parameters.
video_args = {
  'dimensions': 768,
  'region': aoi,
  'framesPerSecond': 7,
  'crs': 'EPSG:3857',
  'min': -40.0,
  'max': 35.0,
  'palette': ['blue', 'purple', 'cyan', 'green', 'yellow', 'red']
}

# Get URL that will produce the animation when accessed.
gif_url = temp_col.getVideoThumbURL(video_args)

# Download animation to Google Drive.
import urllib.request
gif_name = 'ee_collection_animation.gif' # <-- Need to define
urllib.request.urlretrieve(gif_url, gif_name)