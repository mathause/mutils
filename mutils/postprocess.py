import numpy as np
import xarray as xr
import os




def _maybe_recompute_(source_files, dest_files,
                      check_age=False, force=False):


    # force recompute
    if force:
        return True

    if isinstance(source_files, basestring):
        source_files = [source_files]

    if isinstance(dest_files, basestring):
        dest_files = [dest_files]

    # not yet computed
    if _any_file_does_not_exist_(dest_files):
        return True

    # check if source file is newer as dest_file
    if check_age:
        return _source_file_newer_(source_file, dest_file)

    return False


def _any_file_does_not_exist_(fname):
    not_existing = [not os.path.isfile(fn) for fn in fname]
    return np.any(not_existing)

def _source_file_newer_(source_files, dest_files):
    age_source = [os.path.getctime(sf) for sf in source_files]
    age_dest = [os.path.getctime(df) for df in dest_files]

    newest_source = max(age_source)
    oldest_dest = min(age_dest)
    return oldest_source < newest_dest














