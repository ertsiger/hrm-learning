import os.path
import platform
import requests
import subprocess
import sys
import tarfile
from utils.file_utils import copy, mkdir, rm_file, rm_dir


class ILASPInstaller:
    ILASP_LINUX_URL = "https://github.com/ilaspltd/ILASP-releases/releases/download/v4.1.2/ILASP-4.1.2-ubuntu.tar.gz"
    ILASP_OSX_URL = "https://github.com/ilaspltd/ILASP-releases/releases/download/v4.1.2/ILASP-4.1.2-OSX.tar.gz"

    CLINGO_SRC_URL = "https://github.com/potassco/clingo/archive/refs/tags/v5.5.0.tar.gz"

    TMP_TAR_FILE = "tmp.tar.gz"
    TMP_EXTRACTED_TAR_DIR = "tmp"

    def __init__(self):
        path = sys.path[0]
        self._tmp_path = path
        self._bin_path = os.path.join(path, "bin")
        self._lib_path = os.path.join(path, "lib")

    def run(self):
        print("Creating 'bin' and 'lib' folders...")
        mkdir(self._bin_path)
        mkdir(self._lib_path)

        self._install_ilasp()
        self._install_clingo()

    def _install_ilasp(self):
        print("Installing ILASP...")

        if platform.system() == "Darwin":
            url = self.ILASP_OSX_URL
        elif platform.system() == "Linux":
            url = self.ILASP_LINUX_URL
        else:
            raise RuntimeError(f"Error: Unsupported platform '{platform.system()}'.")

        self._install_program(url, self._on_downloaded_ilasp)

    def _on_downloaded_ilasp(self, ilasp_dir):
        copy(os.path.join(ilasp_dir, "ILASP"), self._bin_path)

    def _install_clingo(self):
        print("Installing clingo...")
        self._install_program(self.CLINGO_SRC_URL, self._on_downloaded_clingo)

    def _on_downloaded_clingo(self, clingo_dir):
        clingo_ver_dir = os.path.join(clingo_dir, "clingo-5.5.0")
    
        # Compiling clingo
        subprocess.call([
            "cmake",
            f"-H{clingo_ver_dir}",
            f"-B{clingo_ver_dir}",
            "-DCMAKE_BUILD_TYPE=Release"
        ])
        subprocess.call([
            "cmake",
            "--build",
            clingo_ver_dir
        ])
        
        # Copying the files
        clingo_bin = os.path.join(clingo_ver_dir, "bin")
        copy(os.path.join(clingo_bin, "clingo"), self._bin_path)
        copy(os.path.join(clingo_bin, "libclingo.so"), self._lib_path)
        copy(os.path.join(clingo_bin, "libclingo.so.4"), self._lib_path)
        copy(os.path.join(clingo_bin, "libclingo.so.4.0"), self._lib_path)

    def _install_program(self, url, on_download_completed_callback):
        # Download the compressed file
        resp = requests.get(url)
        with open(self.TMP_TAR_FILE, "wb") as f:
            f.write(resp.content)

        # Extract the compressed file
        tmp_dir = os.path.join(self._tmp_path, self.TMP_EXTRACTED_TAR_DIR)
        tarfile.open(self.TMP_TAR_FILE).extractall(tmp_dir)
        rm_file(self.TMP_TAR_FILE)

        # Perform particular actions for each program
        on_download_completed_callback(tmp_dir)

        # Remove the directory containing the extracted file
        rm_dir(tmp_dir)


if __name__ == "__main__":
    ILASPInstaller().run()
