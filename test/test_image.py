import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch

from lhotse import MonoCut
from lhotse.custom import CustomFieldMixin
from lhotse.image.image import Image
from lhotse.image.io import PillowInMemoryWriter, PillowWriter

# Use importorskip to skip tests if Pillow is not available
PIL = pytest.importorskip("PIL", reason="Image tests require Pillow.")


@pytest.fixture
def sample_image_array():
    """Create a simple RGB test image as numpy array."""
    # Create a 100x100 RGB image
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_path(sample_image_array):
    """Create a temporary image file from the sample array."""
    import PIL.Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = PIL.Image.fromarray(sample_image_array)
        img.save(f.name)
        yield f.name

    # Clean up the temporary file
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def sample_cut_with_image(sample_image_array):
    """Create a MonoCut with a sample image attached to 'custom_image' field."""
    cut = MonoCut(id="cut_with_img", start=0, duration=1, channel=0)
    cut = cut.attach_image("custom_image", sample_image_array)
    return cut


def test_image_from_writer_file(sample_image_array):
    """Test creating an Image manifest from PillowWriter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = PillowWriter(tmpdir)
        image = writer.store_image("img1", sample_image_array)

        assert image.width == 100
        assert image.height == 100
        assert image.shape == (100, 100)
        assert image.storage_type == "pillow_files"
        assert str(tmpdir) in image.storage_path
        assert not image.is_in_memory
        np.testing.assert_array_equal(image.load(), sample_image_array)


def test_image_from_writer_memory(sample_image_array):
    """Test creating an Image manifest from PillowInMemoryWriter."""
    writer = PillowInMemoryWriter()
    image = writer.store_image("img1", sample_image_array)

    assert image.width == 100
    assert image.height == 100
    assert image.shape == (100, 100)
    assert image.storage_type == "pillow_memory"
    assert image.is_in_memory
    np.testing.assert_array_equal(image.load(), sample_image_array)


def test_image_load(sample_image_array):
    """Test loading an image from disk and memory."""
    # Test file-based image
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = PillowWriter(tmpdir)
        image = writer.store_image("img1", sample_image_array)
        loaded_image = image.load()

        assert loaded_image.shape == (100, 100, 3)
        np.testing.assert_array_equal(loaded_image, sample_image_array)

    # Test in-memory image
    writer = PillowInMemoryWriter()
    image = writer.store_image("img1", sample_image_array)
    loaded_image = image.load()

    assert loaded_image.shape == (100, 100, 3)
    np.testing.assert_array_equal(loaded_image, sample_image_array)


def test_image_serialization(sample_image_array):
    """Test Image serialization to/from dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = PillowWriter(tmpdir)
        image = writer.store_image("img1", sample_image_array)

        image_dict = image.to_dict()
        image2 = Image.from_dict(image_dict)

        assert image2.width == image.width
        assert image2.height == image.height
        assert image2.storage_type == image.storage_type
        assert image2.storage_path == image.storage_path
        assert image2.storage_key == image.storage_key


def test_image_move_to_memory(sample_image_array):
    """Test moving an Image from file storage to memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = PillowWriter(tmpdir)
        image = writer.store_image("img1", sample_image_array)
        assert not image.is_in_memory

        # Move to memory
        image_mem = image.move_to_memory()
        assert image_mem.is_in_memory
        assert image_mem.width == image.width
        assert image_mem.height == image.height

        # Load and verify content
        loaded_image = image_mem.load()
        np.testing.assert_array_equal(loaded_image, sample_image_array)


def test_monocut_attach_image(sample_image_array, sample_image_path):
    """Test attaching an image to a MonoCut."""
    cut = MonoCut(id="cut1", start=0, duration=10, channel=0)

    # Test attaching from path
    cut1 = cut.attach_image("img_path", sample_image_path)
    assert cut1.has_custom("img_path")
    assert isinstance(cut1.img_path, Image)
    # Verify it directly references the original file
    assert cut1.img_path.storage_key == Path(sample_image_path).name
    assert cut1.img_path.storage_path == str(Path(sample_image_path).parent)
    assert cut1.img_path.storage_type == "pillow_files"
    # Check that we can still load the image
    loaded_img1 = cut1.load_img_path()
    assert loaded_img1.shape == (100, 100, 3)

    # Test attaching from array
    cut2 = cut.attach_image("img_array", sample_image_array)
    assert cut2.has_custom("img_array")
    assert isinstance(cut2.img_array, Image)
    assert cut2.img_array.storage_type == "pillow_memory"
    loaded_img2 = cut2.load_img_array()
    assert loaded_img2.shape == (100, 100, 3)

    # Test attaching from bytes
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()
    cut3 = cut.attach_image("img_bytes", img_bytes)
    assert cut3.has_custom("img_bytes")
    assert isinstance(cut3.img_bytes, Image)
    assert cut3.img_bytes.storage_type == "pillow_memory"
    loaded_img3 = cut3.load_img_bytes()
    assert loaded_img3.shape == (100, 100, 3)


def test_load_custom_with_image(sample_image_array):
    """Test loading an image via load_custom method."""
    from dataclasses import dataclass

    # Create a simple custom field mixin with an image
    @dataclass
    class CustomObject(CustomFieldMixin):
        custom: Optional[Dict[str, Any]] = None

    # Create test Image object
    writer = PillowInMemoryWriter()
    image = writer.store_image("img1", sample_image_array)

    # Create a custom object with the image
    obj = CustomObject(custom={"test_image": image})

    # Test direct attribute access
    assert obj.test_image is image

    # Test using load_custom method
    loaded_img = obj.load_custom("test_image")
    assert loaded_img.shape == (100, 100, 3)
    np.testing.assert_array_equal(loaded_img, sample_image_array)

    # Test using load_* property
    loaded_img2 = obj.load_test_image()
    assert loaded_img2.shape == (100, 100, 3)
    np.testing.assert_array_equal(loaded_img2, sample_image_array)


def test_monocut_load_custom_with_image(sample_image_array):
    """Test loading an image from a MonoCut using load_custom."""
    cut = MonoCut(id="cut1", start=0, duration=10, channel=0)

    # Test with attach_image
    cut1 = cut.attach_image("test_img", sample_image_array)

    # Test using load_custom method
    loaded_img = cut1.load_custom("test_img")
    assert loaded_img.shape == (100, 100, 3)
    np.testing.assert_array_equal(loaded_img, sample_image_array)

    # Test using load_* property
    loaded_img2 = cut1.load_test_img()
    assert loaded_img2.shape == (100, 100, 3)
    np.testing.assert_array_equal(loaded_img2, sample_image_array)


def test_save_and_load_cut_with_image(sample_image_array):
    """Test loading an image from a MonoCut using load_custom."""
    cut = MonoCut(id="cut1", start=0, duration=10, channel=0)
    cut = cut.attach_image("test_img", sample_image_array)

    cut1 = MonoCut.from_dict(cut.to_dict())

    # Test using load_* property
    loaded_img1 = cut1.load_test_img()
    assert loaded_img1.shape == (100, 100, 3)
    np.testing.assert_array_equal(loaded_img1, sample_image_array)


def test_image_plot(sample_image_array):
    """Test Image.plot() method."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    # Test with in-memory image
    writer = PillowInMemoryWriter()
    image_mem = writer.store_image("img_mem_plot", sample_image_array)

    fig, ax = plt.subplots()
    returned_ax = image_mem.plot(ax=ax)
    assert ax is returned_ax, "The returned Axes should be the one passed as argument."
    # Basic check if something was plotted - check if it has an image
    assert len(ax.images) > 0, "Axes should contain an image after plotting."
    plt.close(fig)  # Close the figure to free memory

    # Test with file-based image and no ax provided
    with tempfile.TemporaryDirectory() as tmpdir:
        writer_file = PillowWriter(tmpdir)
        image = writer_file.store_image("img_file_plot", sample_image_array)

        returned_ax_no_arg = image.plot()
        assert isinstance(
            returned_ax_no_arg, plt.Axes
        ), "Plot should return an Axes object."
        assert (
            len(returned_ax_no_arg.images) > 0
        ), "Axes should contain an image after plotting."
        plt.close(
            returned_ax_no_arg.figure
        )  # Close the figure associated with the returned Axes


def test_collate_images(sample_cut_with_image, sample_image_array):
    """Test collate_images function."""
    from lhotse import CutSet
    from lhotse.dataset.collation import collate_images

    # Create a CutSet with multiple identical cuts for testing collation
    cuts = CutSet.from_cuts([sample_cut_with_image, sample_cut_with_image])

    collated_imgs_tensor = collate_images(cuts, image_field="custom_image")

    assert collated_imgs_tensor.ndim == 4  # (batch, height, width, channel)
    assert collated_imgs_tensor.shape[0] == 2  # Batch size
    assert collated_imgs_tensor.shape[1:] == sample_image_array.shape  # H, W, C
    np.testing.assert_array_equal(
        collated_imgs_tensor[0].cpu().numpy(), sample_image_array
    )
    np.testing.assert_array_equal(
        collated_imgs_tensor[1].cpu().numpy(), sample_image_array
    )


def test_collate_custom_field_with_image(sample_cut_with_image, sample_image_array):
    """Test collate_custom_field for a field containing an Image object."""
    from lhotse import CutSet
    from lhotse.dataset.collation import collate_custom_field

    cuts = CutSet.from_cuts([sample_cut_with_image, sample_cut_with_image])

    # Collate the 'custom_image' field which holds an Image object
    collated_data = collate_custom_field(cuts, field="custom_image")

    assert isinstance(collated_data, torch.Tensor)
    assert collated_data.ndim == 4  # (batch, height, width, channel)
    assert collated_data.shape[0] == 2  # Batch size
    assert collated_data.shape[1:] == sample_image_array.shape  # H, W, C
    np.testing.assert_array_equal(collated_data[0].cpu().numpy(), sample_image_array)
    np.testing.assert_array_equal(collated_data[1].cpu().numpy(), sample_image_array)
