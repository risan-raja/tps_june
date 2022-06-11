from db_store import DB
import warnings

if __name__ =='__main__':
    e = DB()
    with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", message="TBB Warning: The number of workers is currently limited to 3. The request for 31 workers is ignored. Further requests for more workers will be silently ignored until the limit changes.")
        s = e.data_missing_cols(0)
        print(s.shape)
