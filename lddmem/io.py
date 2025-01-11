

def read_field(path, ext):
    """Read a field"""

    if ext in ('.nii', '.gz'):
        import nibabel
        img = nibabel.load(abspath(path))
        img_data = img.get_data().squeeze()
    elif ext == '.nrrd':
        import nrrd
        img_data, img_meta = nrrd.read(abspath(path))
    elif ext in ('.tiff', '.tif'):
        import tifffile
        img_data = tifffile.imread(abspath(path))
        img_data = img_data.transpose(0,2,3,1)
    return img_data


def write_field(field, path, ext):
    """Write a field"""

    if ext in ('.nii', '.gz'):
        import nibabel
        img = nibabel.Nifti1Image(field, np.eye(4))
        nibabel.save(img, path+'.nii.gz')
    elif ext == '.nrrd':
        import nrrd
        nrrd.write(field, path+'.nrrd')
    elif ext in ('.tiff', '.tif'):
        import tifffile
        tifffile.imwrite(
            path+'.tiff',
            field.astype(np.float32).transpose(0,3,1,2),
            imagej=True,
            metadata={'axes':'ZCYX'},
        )
