from ..functions.update_embeddings import update_embeddings
from .commons import Task
import luigi
from . import ingestion
from . import load_model
from pycarol.pipeline import inherit_list
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

@inherit_list(
    ingestion.IngestDocuments,
    load_model.LoadModel
)
class UpdateEmbeddings(Task):

    app_name = luigi.Parameter()
    search_fields = luigi.Parameter()
    keyword_fields = luigi.Parameter()
    refresh_url = luigi.Parameter()
    preproc_mode = luigi.Parameter()
    cache = luigi.BoolParameter()

    def easy_run(self, inputs):

        documents_df = inputs[0]
        model = inputs[1]

        return update_embeddings(
            df=documents_df,
            cols_search=self.search_fields,
            cols_keywords=self.keyword_fields,
            app_name=self.app_name,
            url=self.refresh_url,
            model=model,
            cache=self.cache,
            preproc=self.preproc_mode
        )
