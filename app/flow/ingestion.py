from ..functions.ingestion import data_ingestion
from ..flow.commons import Task
import luigi
import pandas as pd
from pycarol.pipeline import inherit_list
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)
class IngestDocuments(Task):
    staging_name = luigi.Parameter() 

    def easy_run(self, inputs):
        articles = data_ingestion(stag=self.staging_name)
        return articles