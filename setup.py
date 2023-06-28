from setuptools import setup

setup(
    name="overpixel",
    version="0.1.0.dev0",
    author="Edward Evans",
    author_email="elevans2@wisc.edu",
    platforms=["any"],
    entry_points={"console_scripts": ["overpixel=overpixel.__init__:main"]},
    description="A package for analyzing two images for pixel correlation.",
)
