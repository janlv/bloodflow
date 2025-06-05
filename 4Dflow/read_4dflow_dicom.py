#!/usr/bin/env python3

from itertools import groupby
import sys
from pathlib import Path
import numpy as np
from pydicom import dcmread
from tqdm import tqdm
from pyvista import ImageData, read as pvread
from scipy.spatial import KDTree

#MAG = 3  # Number of the magnitude series

#path = Path(sys.argv[1])
#stl_file = len(sys.argv)>2 and Path(sys.argv[2]) or None #'4DFlowBloodpoolEdit1807_1_Edit1807_mesh.stl'
#name = f'4Dflow_{stl_file.stem if stl_file else ""}'

#--------------------------------------------------------------------------------
def distance_map(surface, grid, workers=2, **kwargs):
#--------------------------------------------------------------------------------
    ### dist and idx are the same length as grid_points
    ### dist[0] : Distance from grid_points[0] to mesh_points[idx[0]]
    surface_points = surface.cell_centers().points.astype(np.double)
    grid_points = grid.cell_centers().points.astype(np.double)
    dist, idx = KDTree(surface_points).query(grid_points, workers=workers, **kwargs)
    norm_dot_vec = surface.cell_normals[idx] * (surface_points[idx]-grid_points)
    inside = np.sum(norm_dot_vec, axis=1) > 0
    sign = -np.ones(inside.shape)
    sign[inside] = 1  # Positive values inside surface
    #print(sign.shape, sign.size)
    return sign * dist


#==================================================================
class PVD_file:
#==================================================================
    IND = 2*' '
#------------------------------------------------------------------
    def __init__(self, name, type='Collection') -> None:
#------------------------------------------------------------------
        self.path = Path(name).with_suffix('.pvd')
        self.type = type
        self.file = None
        self.datasets = []

#------------------------------------------------------------------
    def __enter__(self):
#------------------------------------------------------------------
        self.file = open(self.path, 'w')
        print(f'  Creating {self.path}...')
        head = f'<?xml version="1.0"?>\n<VTKFile type="{self.type}">\n{self.IND}<{self.type}>\n'
        self.file.write(head)
        return self

#------------------------------------------------------------------
    def __exit__(self, exc_type, exc_value, traceback):
#------------------------------------------------------------------
        self.file.write(''.join(self.datasets))
        tail = self.IND + f'</{self.type}>\n</VTKFile>\n'
        self.file.write(tail)
        self.file.close()

#------------------------------------------------------------------
    def add_dataset(self, file, time):
#------------------------------------------------------------------
        #print(f'<DataSet timestep="{time:.2f}" part="0" file="{file}"/>')
        ds = 2*self.IND + f'<DataSet timestep="{time:.2f}" part="0" file="{file}"/>\n'
        self.datasets.append(ds)


#------------------------------------------------------------------
def main(path, stl_file=None, cut=None):
#------------------------------------------------------------------
    MAG = 3  # Number of the magnitude series

    # Read dicom-files
    filelist = list(path.rglob('*.dcm'))
    dcmfiles = ((dcmread(f),f) for f in tqdm(filelist, desc='  Reading DICOM files', ncols=100))
    # Remove files that are not valid dicom files
    dcm = [d for d in dcmfiles if hasattr(d[0], 'SliceLocation')]

    print('  Processing data...')
    ### Group dicom by series number
    number = lambda x: x[0].SeriesNumber
    location = lambda x: -x[0].SliceLocation
    series = [list(g) for k,g in groupby(sorted(dcm, key=number), key=number)]
    ### Sort series on instance and location
    series = [sorted(s, key=lambda x: x[0].InstanceNumber) for s in series]
    ### !NB! Better to use cross product of x and y to calculate location !NB!
    series = [sorted(s, key=location) for s in series]
    ### slices[series][slice][time][0=dcm, 1=file]
    slices = [[list(g) for k,g in groupby(s, key=location)] for s in series]
    ### Create array with indices (max): series number (4), time (20), nz (48), 
    S = np.array(slices).transpose(0,2,1,3)[:,:,:,0]
    S0 = S[0,0,0]
    ### Extract pixel_array with indices: time (20), series number (4), nx, ny, nz
    nx, ny, nz = S0.Rows, S0.Columns, S.shape[-1]
    pixels = np.array([s.pixel_array for s in S.flatten()], dtype=np.double).reshape(S.shape + (nx, ny)).transpose(1,0,3,4,2)
    ### Extract time for each slice
    time = np.array([s.TriggerTime for s in S.flatten()]).reshape(S.shape).transpose(1,0,2)

    ### Adjust velocity pixel values
    print('  Adjusting velocity...')
    #venc = np.array((200, 200, 200)) # What is this? venc = velocity encoding 200 cm/s
    venc = 150 * np.ones(3) # What is this? venc = velocity encoding 150 cm/s
    level = 2**S0.HighBit-1
    pixels[:, :MAG, ...] = (pixels[:, :MAG, ...]-level) * venc[None, :, None, None, None] / level

    print('  Transforming data to RCS...')
    # RCS = Reference Coordinate System
    # Contruct 4x4 transformation matrix M
    X, Y = np.split(np.array(S0.ImageOrientationPatient), 2) # pylint: disable=unbalanced-tuple-unpacking
    Z = np.cross(X, Y)
    IPP = np.array([s.ImagePositionPatient for s in S[0,0,:2]])
    shift = np.array(S[0,0,0].ImagePositionPatient)
    M = np.column_stack((X, Y, Z, shift))
    M = np.vstack((M,[0,0,0,1]))
    # Grid resolution
    IPP = np.array([s.ImagePositionPatient for s in S[0,0,:2]])
    (dx, dy), dz = S0.PixelSpacing, np.abs(np.dot(Z, IPP[1]-IPP[0]))
    # Create cell-data grid
    grid = ImageData()
    grid.origin = (0, 0, 0)
    # Note the interchange between x- and y-axis!!!
    grid.dimensions = np.array([ny, nx, nz]) + 1
    grid.spacing = (dy, dx, dz)
    if cut:
        cut += (6-len(cut))*(0,)
        az, bz, by, ay, ax, bx = cut
        print(ax, bx, ay, by, az, bz)
        pixels[:,:,:ax,:,:] = -1
        pixels[:,:,:,:ay,:] = -1
        pixels[:,:,:,:,:az] = -1
        pixels[:,:,-bx or nx:,:,:] = -1 # need nx because [-0:] = whole array
        pixels[:,:,:,-by or ny:,:] = -1
        pixels[:,:,:,:,-bz or nz:] = -1
    # Interchange x- and y-axis and flatten last three dimensions
    pixels = pixels.transpose(0,1,3,2,4).reshape(pixels.shape[:2] + (-1,), order='F')
    # Rotate grid using 4x4 rotation matrix M
    grid = grid.transform(M, inplace=False)

    if stl_file:
        print(f'  Removing data outside vessels using {stl_file}...')
        # Use STL-file to remove data outside the blood vessels
        surface = pvread(stl_file).extract_surface()
        dist_map = distance_map(surface, grid)
        grid['distance'] = dist_map

    # if cut:
    #     grid['cut'] = np.zeros(grid.dimensions)
    #     a, b, c = cut
    #     grid['cut'][a:-a,b:-b,c:-c] = 1

    # Write data as vtk cell-data
    #print('  Saving files...')
    # Loop over times
    stl_str = '_'+Path(stl_file).stem if stl_file else ''
    cut_str = '_'+'x'.join(str(v) for v in cut) if cut else ''
    name = '4Dflow' + stl_str + cut_str
    vel_sum_2 = None
    with PVD_file(path/name) as pvd:
        # NB! The pixel arrays are already flattened for the x-, y-, and z-axis (reshape on line 62)
        for i, series in tqdm(list(enumerate(pixels)), desc='  Writing VTK files', ncols=100):
            vtk = grid.copy()
            vtk.cell_data['magnitude'] = series[MAG]
            vtk.cell_data['velocity'] = np.transpose(np.vstack(series[:MAG]))
            ext = 'vts'
            if cut:
                vtk = vtk.threshold(value=0, scalars='magnitude')
                ext = 'vtu'
            if stl_file:
                vtk = vtk.threshold(value=-10, scalars='distance')
                ext = 'vtu'
            t = time[i][0][0]
            file = path/f'{name}__{i:02d}__{t:03.0f}-ms.{ext}'
            vtk.save(file)
            pvd.add_dataset(file.resolve(), t)
            sum2 = np.sum(vtk.cell_data['velocity']**2, axis=1)
            if vel_sum_2 is None:
                vel_sum_2 = sum2
            else:
                vel_sum_2 += sum2
    # vtk = vtk.copy()
    # print(vtk)
    # print(vel_sum_2.shape)
    # vtk['vel_sum2'] = vel_sum_2
    # vtk.save(path/(name+'__SUM.vtk'))


if __name__ == '__main__':
    sys.argv.pop(0) # Remove script name
    path = Path(sys.argv.pop(0))
    stl_file = None
    cut = []
    for arg in sys.argv:
        if Path(arg).suffix.lower() == '.stl':
            stl_file = arg
        elif arg.isnumeric():
            cut.append(int(arg))
        # else:
        #     raise SystemExit('\nUsage: read_4dflow_dicom.py <dicom-path> [<stl-file> | <cut-x> <cut-y> <cut-z>]\n')
    main(path, stl_file=stl_file, cut=cut)

