from pathlib import Path
from itertools import pairwise, islice #, batched available in 3.12
from pyvista import read as pvread, ImageData, Plotter
from scipy.spatial import KDTree
from numpy import array, ceil, sum as npsum, indices, delete, vstack, sqrt, mean



#====================================================================================
class Surface:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, stl_file, subdivide=0, **kwargs) -> None:
    #--------------------------------------------------------------------------------
        self.stl = Path(stl_file)
        mesh = pvread(Path(stl_file).resolve()).extract_surface()
        if subdivide:
            mesh = mesh.subdivide(subdivide, **kwargs)
        self.min, self.max = [array(mesh.bounds[i::2]) for i in (0,1)]
        self.mesh = mesh

    #--------------------------------------------------------------------------------
    def __str__(self):
    #--------------------------------------------------------------------------------
        return f'{self.stl.name}, min:{self.min}, max:{self.max}, center:{self.center}'

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):
    #--------------------------------------------------------------------------------
        return getattr(self.mesh, item)

    #--------------------------------------------------------------------------------
    def line_segments(self):
    #--------------------------------------------------------------------------------
        ugrid = self.mesh.cast_to_unstructured_grid()
        connect = ugrid.cell_connectivity
        # Array of point-pairs that define a line
        pairs = array([*pairwise(connect)])
        # We only need lines within each cell, but the pairs array
        # also contains lines between cell. The offset array holds the
        # positions of these extra point-pairs that we need to remove.
        # However, we skip the first and last offset entry
        remove_list = ugrid.offset[1:-1]-1
        pairs = delete(pairs, remove_list, axis=0)
        a, b = list(zip(*pairs))
        A = vstack(ugrid.points[a, :])
        B = vstack(ugrid.points[b, :])
        # Calculate length of each line segment
        L = sqrt(npsum((A-B)**2, axis=1))
        return L


#====================================================================================
class Grid:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, surface, dx=None, voxel=(1,1,1), wall=0, echo=False) -> None:
    #--------------------------------------------------------------------------------
        self.surface = surface
        dx = dx or mean(surface.line_segments())
        self.voxel = dx*array(voxel)
        self.wall = wall
        self.echo_on = echo
        self.size = surface.max - surface.min + 2*self.wall*self.voxel
        self.dim = ceil(self.size/self.voxel).astype(int)
        #self.center = [0.5*(a+b) for a,b in batched(surface.bounds, 2)]
        self.ugrid = None
        self.grid = None
        self.distance = 'distance'

    #--------------------------------------------------------------------------------
    def __setitem__(self, key, value):
    #--------------------------------------------------------------------------------
        self.grid[key] = value

    #--------------------------------------------------------------------------------
    def __getitem__(self, key):
    #--------------------------------------------------------------------------------
        return self.grid[key]

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):
    #--------------------------------------------------------------------------------
        return getattr(self.grid, item)

    #--------------------------------------------------------------------------------
    def __str__(self):
    #--------------------------------------------------------------------------------
        return (f'  Dim:   {self.dim}\n'
                f'  Voxel: {self.voxel}\n'
                f'  Size:  {self.size}')

    #--------------------------------------------------------------------------------
    def create(self, threshold=0, method='cells', workers=1, **kwargs):
    #--------------------------------------------------------------------------------
        self.make_uniform_grid()
        self.add_distance_map(coords=method, workers=workers)
        # Remove grid-cells outside surface (negative values)
        self.threshold(value=threshold)
        if self.echo_on:
            print(self)

    #@classmethod
    #--------------------------------------------------------------------------------
    def echo(msg):
    #--------------------------------------------------------------------------------
        # Decorator factory that prints 'msg ... ' and 'done' before and after function calls
        def decorator(func):
            def inner(self, *args, **kwargs):
                if self.echo_on:
                    print(f'  {msg} ... ', end='', flush=True)
                out = func(self, *args, **kwargs)
                if self.echo_on:
                    print('done')
                return out
            return inner
        return decorator

    #--------------------------------------------------------------------------------
    def save(self, path=None, **kwargs):
    #--------------------------------------------------------------------------------
        path = Path(path or self.surface.stl)
        name = f"{path.stem}__{'x'.join(map(str, self.dim))}.vtk"
        path = path.with_name(name)
        self.grid.save(path, **kwargs)
        if self.echo_on:
            print(f'  Saved {path}')

    #--------------------------------------------------------------------------------
    def plot_mesh(self, surface=True, grid=False, show_edges=True, clip=False, **kwargs):
    #--------------------------------------------------------------------------------
        pl = Plotter()
        if surface:
            pl.add_mesh(self.surface.mesh, show_edges=show_edges, opacity=0.7)
        if grid:
            pl.add_mesh(self.ugrid, show_edges=show_edges, opacity=0.7, **kwargs)
        else:
            data = self.grid
            if clip:
                data = self.grid.clip(clip)
            pl.add_mesh(data, show_edges=show_edges, **kwargs)
        pl.show()

    @echo('Thresholding grid')
    #--------------------------------------------------------------------------------
    def threshold(self, value=0, **kwargs):
    #--------------------------------------------------------------------------------
        # if invert:
        #     value = (min(self.min), value)
        self.grid = self.ugrid.threshold(value=value, scalars=self.distance, **kwargs)
        #return self

    @echo('Creating uniform grid')
    #--------------------------------------------------------------------------------
    def make_uniform_grid(self):
    #--------------------------------------------------------------------------------
        # Create uniform grid
        ugrid = ImageData()
        ugrid.spacing = self.voxel
        ugrid.dimensions = self.dim + 1 # Because we want to add cell data (not point data)
        ugrid.origin = self.surface.center - 0.5*(self.voxel*self.dim)
        ugrid.cells = ugrid.cell_centers().points #.astype(npdouble)
        # Add an index array
        ugrid['ijk'] = indices(self.dim).reshape((3,-1), order='F').transpose()
        self.ugrid = ugrid

    @echo('Calculating distance map')
    #--------------------------------------------------------------------------------
    def add_distance_map(self, coords='cells', workers=1, name='distance', ):
    #--------------------------------------------------------------------------------
        """ 
        Add a distance map to the grid. It is the shortest distance from grid 
        cells to mesh surface. 
        
        'coords'  : Use surface cell centers ('cells'), or surface cell 
                    vertices ('points'), default is 'cells'
        'workers' : number of parallel processes, default is 1
        'name'    : name of the distance map scalar, default is 'distance'
        """
        self.distance = name
        if coords == 'points':
            # Use points
            surf_coords = self.surface.points
            surf_normals = self.surface.point_normals
        elif coords == 'cells':
            # Use cells
            surf_coords = self.surface.cell_centers().points
            surf_normals = self.surface.cell_normals
        else:
            raise SyntaxError(f"ERROR in Grid: add_distance_map coords must be 'points' or 'cells', not '{coords}'")
        _, idx = KDTree(surf_coords).query(self.ugrid.cells, workers=workers)
        norm_times_dist = surf_normals[idx] * (surf_coords[idx] - self.ugrid.cells)
        self.ugrid[self.distance] = npsum(norm_times_dist/self.voxel, axis=1)

    #--------------------------------------------------------------------------------
    def index(self):
    #--------------------------------------------------------------------------------
        """ Return grid indices as three separate vectors """
        return self.grid['ijk'].transpose()

    #--------------------------------------------------------------------------------
    def ijk(self):
    #--------------------------------------------------------------------------------
        """ Return the grid index matrix """
        return self.grid['ijk']


