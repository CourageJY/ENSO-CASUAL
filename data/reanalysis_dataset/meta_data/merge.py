import sys
sys.path.append("")

from data.reanalysis_dataset.meta_data.dowanload_cwat_1851_2014 import *
from data.reanalysis_dataset.meta_data.download_pwat_1851_2014 import *
from data.reanalysis_dataset.meta_data.download_rh_1851_2014 import *
from data.reanalysis_dataset.meta_data.download_vwind_1851_2014 import *
from data.reanalysis_dataset.meta_data.download_uwind_1851_2014 import *

if __name__=='__main__':
    download_cwat()

    download_pwat()

    download_rh()

    download_vwind()

    download_uwind()