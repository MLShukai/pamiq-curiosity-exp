from exp.utils import replicate


class TestReplicate:
    """Test cases for the replicate function."""

    def test_replicate_simple_object(self):
        """Test replicating a simple object."""
        obj = {"key": "value"}
        copies = replicate(obj, 3)

        assert len(copies) == 3
        for i, copy_obj in enumerate(copies):
            assert copy_obj == obj
            assert copy_obj is not obj
            assert id(copy_obj) != id(obj)
            # Ensure each copy is unique
            for j in range(i):
                assert copy_obj is not copies[j]
