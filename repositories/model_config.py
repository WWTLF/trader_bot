import json
from models.ml_model_config import MlModelConfig
from db import get_conn


class MLModelConfigRepo:
    def __init__(self, conn):
        self.conn = conn

    def Upsert(self, model_name: str, config: dict, path: str) -> MlModelConfig:
        # conn = get_conn()
        cur = self.conn.cursor()
        cur.execute(
            "insert into ml_model_config(model_name, config, path) values(%s, %s, %s) on conflict (model_name) do update set config = %s, path = %s RETURNING id",
            (model_name, json.dumps(config), path, json.dumps(config), path),
        )
        id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        # conn.close()
        return MlModelConfig(id, model_name, json.dumps(config), path)
    

    def get_by_name(self, model_name: str) -> MlModelConfig:
        # conn = get_conn()
        cur = self.conn.cursor()
        cur.execute("select id, model_name, config, path from ml_model_config where model_name = %s", (model_name,))
        row = cur.fetchone()
        if row is None:
            return None
        id, model_name, config, path = row
        cur.close()
        # conn.close()
        return MlModelConfig(id, model_name, json.loads(config), path)