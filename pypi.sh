python3 setup.py sdist bdist_wheel &&
twine upload dist/*
rm -r build/ dist/ yft.egg-info/
