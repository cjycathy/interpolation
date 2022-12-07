import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from math import radians, cos, sin, asin, sqrt, ceil
import time
from multiprocessingUtils import MultiprocessingUtils

class IdwInterpolation:
    """
    Generates an Inverse Distance Weighted (IDW) interpolation of a point vector layer with parallel computing optimization through multiprocessing module.
    
    Parameters
    ----------
    shp_file           : string 
                         .shp file path of point vector layer

    extent_file        : string 
                         extent of output raster

    resolution         : float
                         Pixel size of the output raster layer in layer units

    max_distance       : integer
                         max distance from the known points to unknown points, known points within the distance are used for idw calculation

    max_point_count    : integer
                         number of known points used for idw calculation, filter base on distance sort from nearest  

    interpolation_field : string
                          field name of the attribute used for interpolation

    process_count      : integer
                         number of processors for calculate

    Examples
    --------
    >>> from interpolation.interpolationUtils import IdwInterpolation
    >>> shp_file = 'sample_point.shp'
    >>> extent_file = 'extent.shp'
    >>> output_tiff = 'output.tif'
    >>> idwInterpolation = IdwInterpolation(shp_file, extent_file, 0.0005, 0.005, 12, 'value', 12)
    >>> idwInterpolation.generate_tiff(output_tiff)
    
    """
    def __init__(self, shp_file, extent_file, resolution, max_distance, max_point_count, interpolation_field, process_count=8):
        self.resolution = resolution
        self.process_count = process_count
        self.interpolation_field = interpolation_field
        self.max_distance = max_distance
        self.max_point_count = max_point_count

        self.extent = self.get_extent_by_shp(extent_file)
        self.grid_shape = self.get_grid_shape()
        self.xyz_np = self.get_shp_data(shp_file)


    def get_extent_by_shp(self, file):
        """
        Returns extent coordinates of a shapefile

        Parameters
        ----------
        extent_file     : .shp file path 
                         extent of output raster 

        Returns
        -------
        array of extent coordinates [min_lon, max_lon, min_lat, max_lat]
        """
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.Open(file, 0)
        extent_list = []
        if data_source is None:
            raise ("can not find extent file")
        else:
            layer = data_source.GetLayer()
            extent = layer.GetExtent()
            for value in extent:
                extent_list.append(value)
        data_source.Destroy()
        return extent_list

    def get_grid_shape(self):
        """
        Returns grid shape from extent and resolution 

        Returns
        -------
        Array of grid hight and width
        """
        min_lon = self.extent[0]
        min_lat = self.extent[2]
        max_lon = self.extent[1]
        max_lat = self.extent[3]
        long_range = sqrt((min_lon - max_lon)**2)
        lat_range = sqrt((min_lat - max_lat)**2)
        grid_width = ceil(long_range/self.resolution)
        grid_height = ceil(lat_range/self.resolution)

        return [grid_height, grid_width]

    def get_shp_data(self, shp_file):
        """
        Returns coordinates and interpolation field name from shapefile

        Parameters
        ----------
        shp_file        : string 
                         .shp file path of point vector layer

        Returns
        -------
        xyz_np          : numpy.ndarray
                         points of x,y coordinates and interpolation field name z, [[x,y,z],...]

        """
        driver = ogr.GetDriverByName('ESRI Shapefile')
        data_source = driver.Open(shp_file, 1)
        layer = data_source.GetLayer(0)
        self.crs_wkt = layer.GetSpatialRef().ExportToWkt()
        xyz_list = []

        for feature in layer:
            x = feature.GetGeometryRef().GetX()
            y = feature.GetGeometryRef().GetY()
            z = 0
            if self.interpolation_field != '':
                z = feature.GetField(self.interpolation_field)
            if z is None:
                z = 0
            xyz_list.append([x, y, z])

        xyz_np = np.asarray(xyz_list)
        data_source.Destroy()
        return xyz_np

    def generate_tiff(self, output_tiff):
        """
        Method that interpolates field of point numpy array to raster file with multiprocessing

        Parameters
        ----------
        output_tiff      : string
                          .tiff file path of output raster
        """
        row_size = self.grid_shape[0]
        column_size = self.grid_shape[1]
        interpolate_np = np.zeros((row_size, column_size))        
        result_list = []
        interpolate_df = pd.DataFrame(interpolate_np)

        m = MultiprocessingUtils(interpolate_df, self.process_count, self.process_idw_by_row)
        result_list = m.run('raster')
        result_list_reverse = np.asarray(result_list)[::-1]
        # self.process_idw_by_df(interpolate_df)
        # result_list_reverse = np.asarray(self.result_list)[::-1]

        self.create_tiff(result_list_reverse, output_tiff)

    def process_idw_by_row(self, row, result_list, params):
        """
        Method that processes idw interpolation base on row of raster grid

        Parameters
        ----------
        row            : pandas.core.series.Series
                         series of raster grid
        
        result_list     : multiprocess.managers.ListProxy
                         list to store grid value

        params         : object
                         common parameters used for calculation during multiprocessing
        """
        row_list = []
        for i in range(row.size):
            index = row.name * row.size + i
            row_i = int(np.floor(index/row.size))
            column_i = index % row.size
            center_loction = self.get_cell_center(row_i, column_i)
            z = self.calculate_z(center_loction)
            row_list.append(z)
        result_list[row.name] = row_list

    def process_idw_by_df(self, df):
        """
        Method that applies idw interpolation to dataframe dataset;
        As a comparison for the multiprocess

        Parameters
        ----------
        df             : pandas.DataFrame
                         dataset of raster grid
        """
        self.result_list = []
        df.apply(self.process_idw, axis=1)

    def process_idw_by_row2(self, row):
        """
        Method that processes idw interpolation base on row of raster grid
        As a comparison for the multiprocess

        Parameters
        ----------
        row            : pandas.core.series.Series
                         series of raster grid

        """
        row_list = []
        for i in range(row.size):
            index = row.name * row.size + i
            row_i = int(np.floor(index/row.size))
            column_i = index % row.size
            center_loction = self.get_cell_center(row_i, column_i)
            z = self.calculate_z(center_loction)
            row_list.append(z)
        self.result_list.append(row_list)

    def get_cell_center(self, row_i, column_i):
        """
        Returns center coordinates from row index and column index of the cell, extent coordinates and resolution

        Parameters
        ----------
        row_i          : integer
                         index of grid row
                         
        column_i       : integer
                         index of grid column

        Returns
        -------
        center_loction : array
                         x,y coordinate of the cell
        """
        x_center = self.extent[0]+(column_i*self.resolution)+(self.resolution/2)
        y_center = self.extent[3]-(row_i*self.resolution)-(self.resolution/2)
        center_loction = [x_center, y_center]
        return center_loction

    def haversine(self, lon1, lat1, lon2, lat2):
        R =  6371000
        dLon = radians(lon2 - lon1)
        dLat = radians(lat2 - lat1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        d = R * c
        return d

    def calculate_z(self, center_loction):
        """
        Calculates value of the cell from numpy array of known points with idw algorithm

        Parameters
        ----------
        center_loction     : array     
                             x,y coordinates of the cell center 

        Returns
        -------
        res                : float 
                             value of the cell 
        """
        distance_list = []
        for point_x_y_z in self.xyz_np:
            point_xi = point_x_y_z[0]
            point_yi = point_x_y_z[1]
            distance = ((point_xi-center_loction[0])**2+(point_yi-center_loction[1])**2)**0.5
            # distance = self.haversine(point_xi, point_yi, center_loction[0], center_loction[1])
            if distance < self.max_distance:
                distance_list.append(distance)
        if len(distance_list) == 0:
            return 0
        else:
            sum_inf = 0
            inverse_distance_list = []
            distance_list = np.array(distance_list)
            distance_sort_list = np.sort(distance_list)[0:self.max_point_count]
            distance_idx_list = np.argsort(distance_list)[0:self.max_point_count]

            for distance_point in distance_sort_list:
                inverse_distance = distance_point**-2
                sum_inf = sum_inf+inverse_distance
                inverse_distance_list.append(inverse_distance)
            sum_up = 0
            for i in range(len(inverse_distance_list)):
                idx = distance_idx_list[i]
                # inverse_distance_list[i] = inverse_distance_list[i]/sum_infnterpolation
                z_i = self.xyz_np[idx, 2]*inverse_distance_list[i]
                sum_up = sum_up + z_i
            if sum_inf == 0:
                res = 0
            else:
                res = sum_up/sum_inf
            return res

    def create_tiff(self, data_grid, output_tiff):
        """
        Method that output tiff file from grid data  

        Paremeters
        ----------
        data_grid       : numpy.ndarray
                         array of the raster cell
                        
        output_tiff     : string
                         .tiff file path of output raster

        """
        origin_x = self.extent[0] - (self.resolution / 2)
        origin_y = self.extent[2] - (self.resolution / 2)

        driver = gdal.GetDriverByName('GTiff')
        data_source = driver.Create(
            output_tiff, data_grid.shape[1], data_grid.shape[0], 1, gdal.GDT_Float32, )

        data_source.SetProjection(self.crs_wkt)
        gt = [origin_x, self.resolution, 0, origin_y, 0, self.resolution]
        data_source.SetGeoTransform(gt)
        outband = data_source.GetRasterBand(1)
        outband.SetStatistics(np.min(data_grid), np.max(data_grid), np.average(data_grid), np.std(data_grid))
        outband.WriteArray(data_grid)
        data_source = None



if __name__ == "__main__":
    startTime = time.time()
    # path_point_xls='data/data_points.xls'
    # shp_file = r'E:\pyqgis\heatmap\data0928_test\point\point_4490_370200.shp'
    # extent_file = "E:/pyqgis/heatmap/area/id_370200.shp"
    shp_file = r'E:\pyqgis\pyQgisProcessingThemeMap\module\interpolation\sample\sample_point.shp'
    extent_file = r"E:\pyqgis\pyQgisProcessingThemeMap\module\interpolation\sample\extent.shp"
    output_tiff = r'E:\pyqgis\pyQgisProcessingThemeMap\module\interpolation\sample\sample.tiff'
    # shp_file = r'E:\pyqgis\heatmap\data0930\point\point_4490_500100_2.shp'
    # extent_file = "E:/pyqgis/heatmap/area/id_500100.shp"
    # output_tiff = r'E:\pyqgis\heatmap\data0930\500100_2.tif'
    # IdwInterpolation = IdwInterpolation(0.001, 0.1, 12)
    IdwInterpolation = IdwInterpolation(shp_file, extent_file, 0.0005, 0.005, 12, 'value', 12)
    # IdwInterpolation.processInterpolation(output_tiff)
    IdwInterpolation.generate_tiff(output_tiff)
    endTime = time.time()
    print("******************processing time: ", endTime-startTime, "s******************")
