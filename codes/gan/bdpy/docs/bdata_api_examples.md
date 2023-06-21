# BData API examples

### Data API

#### Import module and initialization.

    from bdpy import BData

    # Create an empty BData instance
    bdata = BData()

    # Load BData from a file
    bdata = BData('data_file.h5')

#### Load data

    # Load BData from 'data_file.h5'
    bdata.load('data_file.h5')

#### Show data

    # Show 'key' and 'description' of metadata
    bdata.show_meatadata()

    # Get 'value' of the metadata specified by 'key'
    voxel_x = bdata.get_metadata('voxel_x', where='VoxelData')

#### Data extraction

    # Get an array of voxel data in V1
    data_v1 = bdata.select('ROI_V1')  # shape=(M, num voxels in V1)

    # `select` accepts some operators
    data_v1v2 = bdata.select('ROI_V1 + ROI_V2')
    data_hvc = bdata.select('ROI_LOC + ROI_FFA + ROI_PPA - LOC_LVC')

    # Wildcard
    data_visual = data.select('ROI_V*')

    # Get labels ('image_index') in the dataset
    label_a  = bdata.select('image_index')

#### Data creation

    # Add new data
    x = numpy.random.rand(bdata.dataset.shape[0])
    bdata.add(x, 'random_data')

    # Set description of metadata
    bdata.set_metadatadescription('random_data', 'Random data')

    # Save data
    bdata.save('output_file.h5')  # File format is selected automatically by extension. .mat, .h5,and .npy are supported.
