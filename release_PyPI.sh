git add .
git commit -m "Release qcatch v0.2.1"
# git tag v0.2.1          
git push origin main    
# git push origin v0.2.1  


# Build the package locally
python -m build

# Upload the package to PyPI
twine upload dist/*
