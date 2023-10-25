from pathlib import Path
from itertools import pairwise, islice #, batched available in 3.12
from pyvista import read as pvread, ImageData, Plotter
from scipy.spatial import KDTree
from numpy import array, ceil, sum as npsum, indices, delete, vstack, sqrt, mean, stack


#====================================================================================
class Surface:
#====================================================================================
    """
    Extract the pyvista surface from an STL-file 

    Parameters
        stl_file : str or Path
            Path to the STL-file.

        subdivide : int
            Number of subdivisions. Each subdivision creates 4 new triangles, so the number 
            of resulting triangles is nface*4**subdivide where nface is the current number 
            of faces.
        
        subfilter : str, default: 'linear'
            Can be one of the following:
             - 'butterfly'
             - 'loop'
             - 'linear'
        
        progress_bar : bool, default: False
            Display a progress bar for the subdivision routine.
    """
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
        """
        Return the lengths of all line segments of the surface as a numpy array
        """
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
    """
    Create a uniform grid from a Surface object

    Parameters
        surface : Surface
            The surface the grid will be based on.

        dx : float, default: None
            Grid resolution. If None, dx is set to the mean line lengths of the surface

        voxel : tuple of floats, default: (1,1,1)
            Voxel size.
        
        wall : int, default 0
            Add a layer of wall voxels around the geometry.

        echo : bool, default: False
            Notify on progress. 

    """
    #--------------------------------------------------------------------------------
    def __init__(self, surface, dx=None, voxel=(1,1,1), wall=0, echo=False) -> None:
    #--------------------------------------------------------------------------------
        self.surface = surface
        dx = dx or mean(surface.line_segments())
        voxel = array(voxel)
        self.voxel = dx*voxel
        self.skin = 0.5*sqrt(2) # a node touch the surface if the distance less than this
        self.wall = wall
        self.echo_on = echo
        self.size = surface.max - surface.min + 2*self.wall*self.voxel
        self.dim = ceil(self.size/self.voxel).astype(int)
        self.ugrid = None
        self.grid = None

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
    def create(self, threshold=None, coords='points', workers=1, **kwargs):
    #--------------------------------------------------------------------------------
        """
        Generate the grid and add a distance scalar to each cell.
        This scalar holds the shortest distance from the grid cell center 
        to the surface cell center ('cells') or surface cell vertices ('points')
        depending on the coords parameter. The distance scalar is positive 
        inside the surface, and negative outside the surface.

        Parameters
            threshold : float, default: -0.5*sqrt(2)
                Single min value or (min, max) to be used for the data threshold. 
                Grid cell with a distance outside the threshold value is removed.
            
            invert : bool, default: False
                Invert the threshold results. That is, cells that would have been in the 
                output with this option off are excluded, while cells that would have been 
                excluded from the output are included.
            
            coords : str, default: 'points'
                Distance from grid node center to:
                - 'cells'  : surface cell center
                - 'points' : surface cell vertices

            workers : int, default: 1
                Number of workers to use for parallel processing. If -1 is given 
                all CPU threads are used.
        """
        if threshold is None:
            threshold = -self.skin
        self.make_uniform_grid()
        # Distance is negative outside the surface
        self.add_distance_map(coords=coords, workers=workers)
        # Remove grid-cells with values less than threshold
        self.threshold_ugrid(value=threshold, **kwargs)
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
        """
        Save grid as a VTK-file with the STL filename appended with the 
        dimensions of the grid. Extension is '.vtk'
        """
        path = Path(path or self.surface.stl)
        name = f"{path.stem}__{'x'.join(map(str, self.dim))}.vtk"
        path = path.with_name(name)
        self.grid.save(path, **kwargs)
        if self.echo_on:
            print(f'  Saved {path}')

    #--------------------------------------------------------------------------------
    def _plot_data_(self, grid, pl=None, show_edges=True, clip=False, cmap='jet', **kwargs):
    #--------------------------------------------------------------------------------
        show = False
        if pl is None:
            show = True
            pl = Plotter()
        if clip:
            grid = grid.clip(clip)
        pl.add_mesh(grid, show_edges=show_edges, cmap=cmap, **kwargs)
        if show:
            pl.show()
        return pl, grid

    #--------------------------------------------------------------------------------
    def plot_grid(self, *args, wall=False, **kwargs):
    #--------------------------------------------------------------------------------
        grid = self.grid
        if wall:
            grid = self.grid.threshold(value=(-self.skin, self.skin), scalars='distance')
        pl, grid = self._plot_data_(grid, *args, **kwargs)
        if wall:
            pl.add_lines(self.distance_lines(grid), color='cyan', width=3)
        pl.update()
        return pl

    #--------------------------------------------------------------------------------
    def plot_ugrid(self, *args, **kwargs):
    #--------------------------------------------------------------------------------
        pl, _ = self._plot_data_(self.ugrid, *args, **kwargs)
        return pl

    #--------------------------------------------------------------------------------
    def plot_surface(self, *args, **kwargs):
    #--------------------------------------------------------------------------------
        pl, _ = self._plot_data_(self.surface.mesh, *args, **kwargs)
        return pl

    #--------------------------------------------------------------------------------
    def distance_lines(self, grid=None):
    #--------------------------------------------------------------------------------
        """
        Return an array of line segments holding start and end points. 
        The start points (even indices) are the cell centers, and the 
        end points (odd indices) all lie on the surface. The lines are 
        parallel to the local normal vector of the surface.
        """
        if grid is None:
            grid = self.grid
        # Get distance vector scaled by the voxel resolution
        dist_vec = grid['distance'].reshape(-1,1) * grid['normal'] * self.voxel
        # Create array of line segments
        A = grid.cell_centers().points
        lines = stack((A,A), axis=1).reshape(-1,3)
        # Add distance vector to get the end-point of the line
        lines[1::2, :] += dist_vec
        return lines

    @echo('Thresholding grid')
    #--------------------------------------------------------------------------------
    def threshold_ugrid(self, value=0, **kwargs):
    #--------------------------------------------------------------------------------
        """
        Threshold the uniform grid using the pyvista threshold filter. 
        
        See https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.DataSetFilters.threshold.html
        for details  
        """
        self.grid = self.ugrid.threshold(value=value, scalars='distance', **kwargs)

    @echo('Creating uniform grid')
    #--------------------------------------------------------------------------------
    def make_uniform_grid(self):
    #--------------------------------------------------------------------------------
        """
        Create a uniform grid and add and the index matrix 'ijk'
        """
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
    def add_distance_map(self, coords='cells', workers=1):
    #--------------------------------------------------------------------------------
        """ 
        Add a distance map to the grid. It is the shortest distance from grid 
        cells to mesh surface. 
        
        'coords'  : Use surface cell centers ('cells'), or surface cell 
                    vertices ('points'), default is 'cells'
        'workers' : number of parallel processes, default is 1
        """
        if coords not in ('points','cells'):
            raise SyntaxError(f"ERROR in Grid: add_distance_map coords must be 'points' or 'cells', not '{coords}'")
        if coords == 'points':
            # Use points
            surf_coords = self.surface.points
            surf_normals = self.surface.point_normals
        elif coords == 'cells':
            # Use cells
            surf_coords = self.surface.cell_centers().points
            surf_normals = self.surface.cell_normals
        dist, idx = KDTree(surf_coords).query(self.ugrid.cells, workers=workers)
        # KDTree gives the absolute distance from grid-cell to surface-cell.
        # We need to know if the grid-cell is inside or outside the surface.
        # Calculate dist-vector from surface and grid points
        dist_vec = surf_coords[idx] - self.ugrid.cells
        # Distance is the vector product of the surface normal and the distance vector
        self.ugrid['distance'] = npsum((surf_normals[idx] * dist_vec)/self.voxel, axis=1)
        self.ugrid['normal'] = surf_normals[idx]

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


