from pyvista import read as pvread, UniformGrid, UnstructuredGrid
from scipy.spatial import KDTree
from numpy import array, hsplit, ceil, double as npdouble, sum as npsum, ones
from pathlib import Path

#====================================================================================
class Grid:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, dx=None, stl=None, echo=True) -> None:
    #--------------------------------------------------------------------------------
        self.dx = dx 
        self.grid = None
        #self.solid = {}
        self._surface = None
        self.min, self.max = 0, 0
        self.size = 0
        self.dim = 0
        self.wall = 0
        self.echo = echo
        if stl:
            self._surface = self._get_surface(stl)
            self.min, self.max = hsplit(array(self._surface.bounds).reshape((3,2)), 2)
            self.size = (self.max - self.min).squeeze()
            self.size += 2*self.wall*self.dx
            self.dim = ceil(self.size/self.dx).astype(int)


    #--------------------------------------------------------------------------------
    def _get_surface(self, stl):
    #--------------------------------------------------------------------------------
        stl = Path(stl).resolve()
        if self.echo:
            print(f'  Extracting surface from STL-file {stl.relative_to(Path.cwd())} ... ', end='', flush=True)
        surface = pvread(stl).extract_surface()
        if self.echo:
            print('done')
        return surface

    #--------------------------------------------------------------------------------
    def save(self, **kwargs):
    #--------------------------------------------------------------------------------
        # Filename
        dx_str = f'{int(self.dx)}-{round((self.dx%1)*100):2d}'
        dim_str = f'{"x".join([str(d) for d in self.dim])}'
        file = Path(f'{self._surface.parent/self._surface.stem}__LB__dx_{dx_str}__dim_{dim_str}.vtk')
        self.grid.save(file, **kwargs)
        if self.echo:
            print(f'  Saved {file}')

    #--------------------------------------------------------------------------------
    def make(self, **kwargs):
    #--------------------------------------------------------------------------------
        grid =  self._uniform_grid()
        distance = self._distance_map(grid, **kwargs)
        # Add distance from grid node to surface as cell_data
        grid['distance'] = distance/self.dx
        grid['outside'] = array(grid['distance'] < 0, dtype=int)


    #--------------------------------------------------------------------------------
    def _uniform_grid(self):
    #--------------------------------------------------------------------------------
        # Create uniform grid
        if self.echo:
            print(f'  Creating uniform grid of dimension {self.dim} ...')
        grid = UniformGrid()
        grid.dimensions = self.dim + 1 # Because we want to add cell data (not point data)
        grid.origin = self.min - self.wall*self.dx
        grid.spacing = (self.dx, self.dx, self.dx) # Evenly spaced grids
        grid.cells = grid.cell_centers().points.astype(npdouble)
        #ijk = ( (grid.cells-grid.origin)/dx ).astype(int)
        #grid['IJK'] = ijk
        return grid

    #--------------------------------------------------------------------------------
    def _distance_map(self, grid, workers=1):
    #--------------------------------------------------------------------------------
        # Return shortest distance from grid-cell to mesh surface cell together with mesh index
        surf_cells = self._surface.cell_centers().points.astype(npdouble)
        surf_normals = self._surface.cell_normals
        if self.echo:
            print(f'  Calculating distance from surface to grid nodes using {workers} workers ... ')
        dist, idx = KDTree(surf_cells).query(grid.cells, workers=workers)
        if self.echo:
            print('done')
        norm_dot_vec = surf_normals * (surf_cells[idx] - grid.cells)
        inside = npsum(norm_dot_vec, axis=1) > 0
        sign = -ones(inside.shape)
        sign[inside] = 1  # Positive values inside surface 
        return sign * dist

