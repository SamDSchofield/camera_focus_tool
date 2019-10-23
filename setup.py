from catkin_pkg.python_setup import generate_distutils_setup
from distutils.core import setup

setup_args = generate_distutils_setup(
    packages=["camera_focus_tool"],
    package_dir={
        "camera_focus_tool": 'src/'
    }
)

setup(**setup_args)