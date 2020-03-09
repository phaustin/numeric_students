"""
  a500.utils.data_read 
  ___________________

  downloads a file named filename from the atsc301 downloads directory
  and save it as a local file with the same name. 

  to run from the command line::

    python -m a500.utils.data_read photon_data.csv

    or

    python -m a500.utils.data_read *par_9km.nc --root https://oceandata.sci.gsfc.nasa.gov/cgi/getfile --dest_folder=.

  to run from a python script::

    from a500.utils.data_read import download
    download('photon_data.csv')

  or

    from a500.utils.data_read import download
    root="https://oceandata.sci.gsfc.nasa.gov/cgi/getfile"
    filename="A20162092016216.L3m_8D_PAR_par_9km.nc"
    download(filename,root=root)

"""
import argparse
import requests
from pathlib import Path
import shutil

class NoDataException(Exception):
    pass



def download(
    filename,
    root="https://clouds.eos.ubc.ca/~phil/courses/atsc301/downloads",
    dest_folder=None,
):
    """
    copy file filename from http://clouds.eos.ubc.ca/~phil/courses/atsc301/downloads to 
    the local directory.  If local file exists, report file size and quit.

    Parameters
    ----------

    filename: string
      name of file to fetch from 

    root: optional string 
          to specifiy a different download url

    dest_folder: optional string or Path object
          to specifify a folder besides the current folder to put the files
          will be created it it doesn't exist

    Returns
    -------

    Side effect: Creates a copy of that file in the local directory
    """
    filename = Path(filename)
    name_only = filename.name
    url = f"{root}/{name_only}"
    url = url.replace("\\", "/")
    print("trying {}".format(url))
    #
    # use current directory if dest_dir not specified
    #
    if dest_folder is None:
        dest_path = Path()
    else:
        dest_path = Path(dest_folder).resolve()
        dest_path.mkdir(parents=True, exist_ok=True)
    #
    # filename may contain subfolders
    #
    filepath = dest_path / Path(filename.name)
    print(f"writing to: {filepath}")
    if filepath.exists():
        the_size = filepath.stat().st_size
        print(
            ("\n{} already exists\n" "and is {} bytes\n" "will not overwrite\n").format(
                filename, the_size
            )
        )
        return None

    tempfile = str(filepath) + "_tmp"
    temppath = Path(tempfile)
    try:
        with open(temppath, "wb") as localfile:
            print(f"writing temporary file {temppath}")
            response = requests.get(url, stream=True)
            #
            # treat a 'Not Found' response differently, since you want to catch
            # this and possibly continue with a new file
            #
            if not response.ok:
                if response.reason == "Not Found":
                    the_msg = 'requests.get() returned "Not found" with filename {}'.format(
                        filename
                    )
                    raise NoDataException(the_msg)
                else:
                    #
                    # if we get some other response, raise a general exception
                    #
                    the_msg = "requests.get() returned {} with filename {}".format(
                        response.reason, filename
                    )
                    raise RuntimeError(the_msg)
                    #
                # clean up the temporary file
                #
            for block in response.iter_content(1024):
                if not block:
                    break
                localfile.write(block)
        the_size = temppath.stat().st_size
        print("downloaded {}\nsize = {}".format(filename, the_size))
        shutil.move(str(temppath), str(filepath))
        if the_size < 10.0e3:
            print(
                "Warning -- your file is tiny (smaller than 10 Kbyte)\nDid something go wrong?"
            )
    except NoDataException as e:
        print(e)
        print("clean up: removing {}".format(temppath))
        temppath.unlink()
    return None


def make_parser():
    """
    set up the command line arguments needed to call the program
    """
    linebreaks = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=linebreaks, description=__doc__.lstrip()
    )
    parser.add_argument("filename", type=str, help="name of file to download")
    parser.add_argument(
        "--root",
        default="https://clouds.eos.ubc.ca/~phil/courses/atsc301/downloads",
        help="root of url, detaults to https://clouds.eos.ubc.ca/~phil/courses/atsc301/downloads",
    )
    return parser


def main(args=None):
    parser = make_parser()
    args = parser.parse_args(args)
    download(args.filename, root=args.root)


if __name__ == "__main__":
    main()
