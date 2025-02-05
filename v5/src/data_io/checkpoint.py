import os
import h5py
from datetime import datetime


class CheckpointHandler:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, state: dict, step: int) -> str:
        filename = f"{self.output_dir}/checkpoint_{step:06d}.h5"
        with h5py.File(filename, "w") as f:
            # メタデータ
            f.attrs["time"] = state["time"]
            f.attrs["step"] = state["step"]
            f.attrs["date"] = str(datetime.now())

            # 場の保存
            fields = f.create_group("fields")
            fields.create_dataset("phi", data=state["fields"]["phi"])
            fields.create_dataset("pressure", data=state["fields"]["pressure"])

            # 速度場の保存
            velocity = fields.create_group("velocity")
            for i, v in enumerate(state["fields"]["velocity"]):
                velocity.create_dataset(f"component_{i}", data=v)

        return filename

    def load(self, filename: str) -> dict:
        with h5py.File(filename, "r") as f:
            state = {
                "time": f.attrs["time"],
                "step": f.attrs["step"],
                "fields": {
                    "phi": f["fields/phi"][:],
                    "pressure": f["fields/pressure"][:],
                    "velocity": [
                        f[f"fields/velocity/component_{i}"][:] for i in range(3)
                    ],
                },
            }
        return state
