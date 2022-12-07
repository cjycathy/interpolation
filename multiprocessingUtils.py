import math
from tqdm import tqdm
import multiprocess

class Multiprocessing:
    """
    Processes dataset with function parameter in multiprocess

    Parameters
    ----------     
    dataset:           : pandas.DataFrame
                         dataset to be chunked for multiprocess

    process_count      : integer
                         number of processors for calculate

    process_func:      : callback function
                         function used to process chunked dataset in each process

    params             : object
                         common parameters used for calculation during multiprocessing

    Examples
    --------
    >>> from interpolation.multiprocessingUtils import MultiprocessingUtils
    >>> idw_np = np.zeros((100, 100))
    >>> dataset = pd.DataFrame(idw_np) 
    >>> process_count = 12
    >>> def func(row, result_list):
    ...     row_list = []
    ...     for i in range(row.size):
    ...         row_list.append(100)
    ...     result_list[row.name] = row_list
    >>> m = MultiprocessingUtils(dataset, process_count, func)
    >>> result_list = m.run('raster')

    """

    def __init__(self, dataset, process_count, process_func, params={}):
        self.dataset = dataset
        self.process_count = process_count
        self.process_func = process_func
        self.params = params
        
    def run(self, type):
        """
        Method that generates result list and construct multiprocess pool

        Parameters
        ----------
        type:      : string
                     type of result list value stand for;
                     available options:
                     'raster'
                     'vector'

        """
        manager = multiprocess.Manager()
    
        if type == 'raster':    
            if "result_shape" in self.params:
                result_shape = self.params["result_shape"]
            else:
                result_shape = [self.dataset.shape[0], self.dataset.shape[1]]
            # shared result in all processes used for raster dataset, with specific shape 
            self.result_list = manager.list([[0 for i in range(result_shape[1])] for j in range(result_shape[0])])
        else:   
            # shared result in all processes used for vector dataset 
            self.result_list = manager.list()
 
        with multiprocess.Pool(processes=self.process_count) as p:
            r = list(tqdm(p.imap(self.func, range(self.process_count)), total=self.process_count))
        
        return list(self.result_list)
            
        
    def func(self, i):
        """
        Method that chunks the dataset and applys function for each row of
        dataset according to process count and process index

        Parameters
        ----------
        i      : integer
                 index of process 
        """
        row = self.dataset.shape[0]
        interval = row / self.process_count

        if i < self.process_count-1:
            data = self.dataset[math.floor(interval * i):math.floor(interval * (i+1))]
        else:
            data = self.dataset[math.floor(interval * i):math.floor(interval * (i+1))+1]
        
        data.apply(self.process_func, axis=1, args=(self.result_list, self.params))


