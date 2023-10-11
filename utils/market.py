import os
import copy
from shutil import copyfile, rmtree

import zipfile
from typing import Tuple, Any, List, Union, Dict

from learnware.market.base import BaseMarket, BaseUserInfo
from learnware.market.database_ops import DatabaseOperations

from learnware.learnware import Learnware, get_learnware_from_dirpath
from learnware.specification import RKMEStatSpecification, Specification
from learnware.logger import get_module_logger
from learnware.config import C as conf

logger = get_module_logger("market", "INFO")


class DummyMarket:
    INVALID_LEARNWARE = -1
    NONUSABLE_LEARNWARE = 0
    USABLE_LEARWARE = 1

    def __init__(self, market_id: str = "default", rebuild: bool = False):
        """Initialize Learnware Market.
        Automatically reload from db if available.
        Build an empty db otherwise.

        Parameters
        ----------
        market_id : str, optional, by default 'default'
            The unique market id for market database

        rebuild : bool, optional
            Clear current database if set to True, by default False
            !!! Do NOT set to True unless highly necessary !!!
        """
        self.market_id = market_id
        self.market_store_path = os.path.join(conf.market_root_path, self.market_id)
        self.learnware_pool_path = os.path.join(self.market_store_path, "learnware_pool")
        self.learnware_zip_pool_path = os.path.join(self.learnware_pool_path, "zips")
        self.learnware_folder_pool_path = os.path.join(self.learnware_pool_path, "unzipped_learnwares")
        self.learnware_list = {}  # id: Learnware
        self.learnware_zip_list = {}
        self.learnware_folder_list = {}
        self.count = 0
        self.semantic_spec_list = conf.semantic_specs
        self.dbops = DatabaseOperations(conf.database_url, 'market_' + self.market_id)
        self.reload_market(rebuild=rebuild)  # Automatically reload the market
        logger.info("Market Initialized!")

    def reload_market(self, rebuild: bool = False) -> bool:
        if rebuild:
            logger.warning("Warning! You are trying to clear current database!")
            try:
                self.dbops.clear_learnware_table()
                rmtree(self.learnware_pool_path)
            except:
                pass

        os.makedirs(self.learnware_pool_path, exist_ok=True)
        os.makedirs(self.learnware_zip_pool_path, exist_ok=True)
        os.makedirs(self.learnware_folder_pool_path, exist_ok=True)
        self.learnware_list, self.learnware_zip_list, self.learnware_folder_list, self.count = self.dbops.load_market()

    @classmethod
    def check_learnware(cls, learnware: Learnware) -> int:
        return cls.USABLE_LEARWARE

    def add_learnware(self, zip_path: str, semantic_spec: dict) -> Tuple[str, int]:
        semantic_spec = copy.deepcopy(semantic_spec)

        if not os.path.exists(zip_path):
            logger.warning("Zip Path NOT Found! Fail to add learnware.")
            raise Exception("INVALID_LEARNWARE")

        logger.info("Get new learnware from %s" % zip_path)
        id = "%08d" % self.count
        target_zip_dir = os.path.join(self.learnware_zip_pool_path, "%s.zip" % id)
        target_folder_dir = os.path.join(self.learnware_folder_pool_path, id)
        copyfile(zip_path, target_zip_dir)

        with zipfile.ZipFile(target_zip_dir, "r") as z_file:
            z_file.extractall(target_folder_dir)
        logger.info("Learnware move to %s, and unzip to %s" % (target_zip_dir, target_folder_dir))

        try:
            new_learnware = get_learnware_from_dirpath(
                id=id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
            )
        except:
            try:
                os.remove(target_zip_dir)
                rmtree(target_folder_dir)
            except:
                pass
            raise Exception("INVALID_LEARNWARE")

        if new_learnware is None:
            raise Exception("INVALID_LEARNWARE")

        check_flag = self.check_learnware(new_learnware)

        self.dbops.add_learnware(
            id=id,
            semantic_spec=semantic_spec,
            zip_path=target_zip_dir,
            folder_path=target_folder_dir,
            use_flag=check_flag,
        )

        self.learnware_list[id] = new_learnware
        self.learnware_zip_list[id] = target_zip_dir
        self.learnware_folder_list[id] = target_folder_dir
        self.count += 1
        return id, check_flag

    @staticmethod
    def _search_by_rkme_spec(
            learnware_list: List[Learnware],
            user_rkme: RKMEStatSpecification,
            on: str
    ) -> Tuple[List[float], List[Learnware]]:

        RKME_list = [
            learnware.specification.get_stat_spec_by_name("RKMEStatSpecification") for learnware in learnware_list
        ]

        score_fn, order = {"similarity": (user_rkme.inner_prod, True),
                           "dist": (user_rkme.dist, False)}[on]

        score_list = [score_fn(learnware_rkme) for learnware_rkme in RKME_list]
        sorted_idx = sorted(range(len(learnware_list)),
                            key=lambda k: score_list[k], reverse=order)
        sorted_score = [score_list[idx] for idx in sorted_idx]
        sorted_learnware = [learnware_list[idx] for idx in sorted_idx]

        return sorted_score, sorted_learnware

    def search_learnware(
            self, user_info: BaseUserInfo, max_search_num: int = 5, on="dist"
    ) -> Tuple[List[float], List[Learnware], List[float], List[Learnware]]:

        learnware_list = [self.learnware_list[key] for key in self.learnware_list]

        if "RKMEStatSpecification" not in user_info.stat_info:
            raise Exception("RKMEStatSpecification not in user_info.stat_info")
        elif len(learnware_list) == 0:
            return [], [], [], []
        else:
            user_rkme = user_info.stat_info["RKMEStatSpecification"]

            sorted_score, sorted_learnware = self._search_by_rkme_spec(
                learnware_list, user_rkme, on=on)

            return sorted_score[:max_search_num], sorted_learnware[:max_search_num], [], []

    def delete_learnware(self, id: str) -> bool:
        if not id in self.learnware_list:
            logger.warning("Learnware id:'{}' NOT Found!".format(id))
            return False

        zip_dir = self.learnware_zip_list[id]
        os.remove(zip_dir)
        folder_dir = self.learnware_folder_list[id]
        rmtree(folder_dir)
        self.learnware_list.pop(id)
        self.learnware_zip_list.pop(id)
        self.learnware_folder_list.pop(id)
        self.dbops.delete_learnware(id=id)

        return True

    def get_semantic_spec_list(self) -> dict:
        return self.semantic_spec_list

    def get_learnware_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        if isinstance(ids, list):
            ret = []
            for id in ids:
                if id in self.learnware_list:
                    ret.append(self.learnware_list[id])
                else:
                    logger.warning("Learnware ID '%s' NOT Found!" % (id))
                    ret.append(None)
            return ret
        else:
            try:
                return self.learnware_list[ids]
            except:
                logger.warning("Learnware ID '%s' NOT Found!" % (ids))
                return None

    def get_learnware_path_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        if isinstance(ids, list):
            ret = []
            for id in ids:
                if id in self.learnware_zip_list:
                    ret.append(self.learnware_zip_list[id])
                else:
                    logger.warning("Learnware ID '%s' NOT Found!" % (id))
                    ret.append(None)
            return ret
        else:
            try:
                return self.learnware_zip_list[ids]
            except:
                logger.warning("Learnware ID '%s' NOT Found!" % (ids))
                return None

    def __len__(self):
        return len(self.learnware_list.keys())

    def _get_ids(self, top=None):
        if top is None:
            return list(self.learnware_list.keys())
        else:
            return list(self.learnware_list.keys())[:top]
