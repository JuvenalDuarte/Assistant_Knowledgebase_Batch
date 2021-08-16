import luigi

luigi.interface.InterfaceLogging.setup(luigi.interface.core())

import os
import logging
import traceback

logger = logging.getLogger(__name__)

from pycarol.pipeline import Task
from luigi import Parameter
from datetime import datetime
from pycarol import Carol
from pycarol.apps import Apps

PROJECT_PATH = os.getcwd()
TARGET_PATH = os.path.join(PROJECT_PATH, 'luigi_targets')
Task.TARGET_DIR = TARGET_PATH
#Change here to save targets locally.
Task.is_cloud_target = True
Task.version = Parameter()
Task.resources = {'cpu': 1}

now = datetime.now()
now_str = now.isoformat()
_settings = Apps(Carol()).get_settings()

#================================================================================
# Settings describing the knowledge base
#================================================================================

# This setting specify the table which will be used as the knowledge base
# Usage:
#       - expects {organization}/{environment}/{connector}/{staging}
#       - organization and environment levels may be supressed, assuming the
#           current working environment
#       - connector and staging are mandatory
staging_name = _settings.get('kb_in_staging')

# Specifies which columns from the knowledge base staging will be used 
# on the searchs.
# Usage:
#       - empty: NOT ALLOWED. At least one searcheable field must be provided.
#       - "attribute": a single attribute will be used for seach
#       - ["attr1", "attr2"]: allows a list of fields to be used for search
search_fields = _settings.get('kb_search_fields')

# Additionally to the search fields, it is allowed to offer keywords columns
# for search such as tags, labels etc, which can improve search results.
# Usage:
#       - empty: keywords will not be considered for search
#       - "attribute": a single attribute will be used for keywords seach
#       - ["attr1", "attr2"]: allows a list of fields to be used for keyword search
keyword_fields = _settings.get('kb_keywords_fields')


# Defines if search fields should receive any kind of transformation before
# flowing to the database. In some cases preprocessing steps may increase model
# accuracy and lead the more relevant results.
# Usage:
#       - "Basic": only fix encoding problems
#       - "Advanced": fix encodings, remove special characters, set lowercase 
#                     and remove stopwords.
preproc_mode = _settings.get('preproc_mode')

#================================================================================
# Settings about where and how the knowledge base will be published
#================================================================================

# App where the knowledge base will be saved (as a dafarame). In general, this 
# should point to the online app answering REST requests.
app_name = _settings.get('online_app_name')

# Optional, if the app provides an refresh URL, this URL can be called after 
# data publication, letting the online app knows a knew version of the knowledge
# base is available.
refresh_url = _settings.get('online_app_refreshurl')

#================================================================================
# Optimization parameters
#================================================================================

# Allows the process to reuse embeddings calculated on previous execution, leading 
# to faster executions. If false the full database will be reprocessed.
embeddings_cache = _settings.get('embeddings_cache')

#================================================================================
# Model options
#================================================================================

# If this parameter is filled the app will look for a file with given name
# on its local storage.
model_storage_file = _settings.get('model_storage_file')

# If model_storage_file is empty the app will look for a model with the given 
# name on the sentencetransformers online store.
model_sentencetransformers = _settings.get('model_sentencetransformers')


@Task.event_handler(luigi.Event.FAILURE)
def mourn_failure(task, exception):
    """Will be called directly after a failed execution
       of `run` on any JobTask subclass
    """
    logger.error(f'Error msg: {exception} ---- Error: Task {task},')
    traceback_str = ''.join(traceback.format_tb(exception.__traceback__))
    logger.error(traceback_str)


@Task.event_handler(luigi.Event.PROCESSING_TIME)
def print_execution_time(self, processing_time):
    logger.debug(f'### PROCESSING TIME {processing_time}s. Output saved at {self.output().path}')


#######################################################################################################

params = dict(
    version=os.environ.get('CAROLAPPVERSION', 'dev'),
    datetime = now_str,

    staging_name = staging_name,
    search_fields = search_fields,
    keyword_fields = keyword_fields,
    preproc_mode = preproc_mode,

    app_name=app_name,
    refresh_url = refresh_url,

    model_storage_file = model_storage_file,
    model_sentencetransformers = model_sentencetransformers,

    cache = embeddings_cache
)
