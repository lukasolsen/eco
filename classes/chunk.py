import typing

# Define the structure for a ChunkType
ChunkType = typing.Dict[str, typing.Union[str, typing.List[typing.List[str]]]]

class Chunk:
    def __init__(self, chunk_x: int, chunk_y: int, chunk_size: int = 16):
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.chunk_size = chunk_size
        self.chunk_data = [["empty" for _ in range(chunk_size)] for _ in range(chunk_size)]

    def build_chunk(self, chunk_type: ChunkType):
        """Builds the chunk using the given chunk_type data."""
        self.chunk_data = chunk_type["data"]

    def get_data(self) -> list[list[str]] | str:
        """Returns the chunk's data."""
        return self.chunk_data

    def get_position(self) -> typing.Tuple[int, int]:
        """Returns the chunk's coordinates in the world."""
        return self.chunk_x, self.chunk_y
