from pycarol import Carol, DataModel, Staging, Apps, PwdAuth
import os
import logging
from datetime import datetime, timedelta
import gc
import pandas as pd

logger = logging.getLogger(__name__)

def fetchFromCarol(org, env, conn, stag, columns=None):
    carol = Carol()
    carol.switch_environment(org_name=org, env_name=env, app_name='zendeskdata')

    try:
        df = Staging(carol).fetch_parquet(staging_name=stag, connector_name=conn, backend='pandas', columns=columns, cds=True)

    except Exception as e:
        logger.error(f'Failed to fetch dada. {e}')
        df =  pd.DataFrame()

    return df

def data_ingestion(stag):

    logger.info(f'Parsing \"in_staging\" setting.')

    # Parsing details about the training table connection parameters
    stag_list = stag.split("/")
    if len(stag_list) == 4:
        org, env, conn, stag = stag_list
    elif len(stag_list) == 3:
        env, conn, stag = stag_list
        org = "totvs"
    elif len(stag_list) == 2:
        conn, stag = stag_list
        org = "totvs"
        env = "sentencesimilarity"
    else:
        raise "Unable to parse \"in_staging\" setting. Valid options are: 1. org/env/conn/stag; 2. env/conn/stag; 3. conn/stag."

    logger.info(f'Retrieving data from {org}/{env}/{conn}/{stag}.')
    kcs_df = fetchFromCarol(org=org, env=env, conn=conn, stag=stag)

    logger.info(f'Data ingestion concluded.')

    return kcs_df
