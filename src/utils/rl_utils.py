import gym
from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldTasks
from gym_hierarchical_subgoal_automata.envs.waterworld.waterworld_env import WaterWorldTasks

ENV_PREFIX = "gym_hierarchical_subgoal_automata:"
ENV_CLASSES = {
    CraftWorldTasks.CHICKEN.value: "CraftWorldGetChicken-v0",
    CraftWorldTasks.COW.value: "CraftWorldGetCow-v0",
    CraftWorldTasks.IRON.value: "CraftWorldGetIron-v0",
    CraftWorldTasks.LAVA.value: "CraftWorldGetLava-v0",
    CraftWorldTasks.RABBIT.value: "CraftWorldGetRabbit-v0",
    CraftWorldTasks.REDSTONE.value: "CraftWorldGetRedstone-v0",
    CraftWorldTasks.SQUID.value: "CraftWorldGetSquid-v0",
    CraftWorldTasks.SUGARCANE.value: "CraftWorldGetSugarcane-v0",
    CraftWorldTasks.TABLE.value: "CraftWorldGetTable-v0",
    CraftWorldTasks.WHEAT.value: "CraftWorldGetWheat-v0",
    CraftWorldTasks.WORKBENCH.value: "CraftWorldGetWorkbench-v0",
    CraftWorldTasks.BATTER.value: "CraftWorldBatter-v0",
    CraftWorldTasks.BUCKET.value: "CraftWorldBucket-v0",
    CraftWorldTasks.COMPASS.value: "CraftWorldCompass-v0",
    CraftWorldTasks.LEATHER.value: "CraftWorldLeather-v0",
    CraftWorldTasks.PAPER.value: "CraftWorldPaper-v0",
    CraftWorldTasks.QUILL.value: "CraftWorldQuill-v0",
    CraftWorldTasks.SUGAR.value: "CraftWorldSugar-v0",
    CraftWorldTasks.BOOK.value: "CraftWorldBook-v0",
    CraftWorldTasks.MAP.value: "CraftWorldMap-v0",
    CraftWorldTasks.MILK_BUCKET.value: "CraftWorldMilkBucket-v0",
    CraftWorldTasks.BOOK_AND_QUILL.value: "CraftWorldBookAndQuill-v0",
    CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value: "CraftWorldMilkBucketAndSugar-v0",
    CraftWorldTasks.CAKE.value: "CraftWorldCake-v0",
    CraftWorldTasks.TEST_LOOP.value: "CraftWorldTestLoop-v0",
    CraftWorldTasks.TEST_CONTEXT.value: "CraftWorldTestContext-v0",
    CraftWorldTasks.TEST_DISJUNCTION.value: "CraftWorldTestDisjunction-v0",
    CraftWorldTasks.TEST_DISJUNCTION_SIMPLE.value: "CraftWorldTestDisjunctionSimple-v0",
    CraftWorldTasks.TEST_DISJUNCTION_DOUBLE.value: "CraftWorldTestDisjunctionDouble-v0",
    CraftWorldTasks.TEST_SIMULTANEOUS_SAT.value: "CraftWorldTestSimultaneousSat-v0",
    CraftWorldTasks.TEN_PAPERS.value: "CraftWorldTenPapers-v0",
    WaterWorldTasks.R.value: "WaterWorldR-v0",
    WaterWorldTasks.G.value: "WaterWorldG-v0",
    WaterWorldTasks.B.value: "WaterWorldB-v0",
    WaterWorldTasks.C.value: "WaterWorldC-v0",
    WaterWorldTasks.M.value: "WaterWorldM-v0",
    WaterWorldTasks.Y.value: "WaterWorldY-v0",
    WaterWorldTasks.RG.value: "WaterWorldRG-v0",
    WaterWorldTasks.BC.value: "WaterWorldBC-v0",
    WaterWorldTasks.MY.value: "WaterWorldMY-v0",
    WaterWorldTasks.RE.value: "WaterWorldRE-v0",
    WaterWorldTasks.GE.value: "WaterWorldGE-v0",
    WaterWorldTasks.BE.value: "WaterWorldBE-v0",
    WaterWorldTasks.CE.value: "WaterWorldCE-v0",
    WaterWorldTasks.YE.value: "WaterWorldYE-v0",
    WaterWorldTasks.ME.value: "WaterWorldME-v0",
    WaterWorldTasks.RGB.value: "WaterWorldRGB-v0",
    WaterWorldTasks.CMY.value: "WaterWorldCMY-v0",
    WaterWorldTasks.RG_BC.value: "WaterWorldRGAndBC-v0",
    WaterWorldTasks.BC_MY.value: "WaterWorldBCAndMY-v0",
    WaterWorldTasks.RG_MY.value: "WaterWorldRGAndMY-v0",
    WaterWorldTasks.RGB_CMY.value: "WaterWorldRGBAndCMY-v0",
    WaterWorldTasks.RG_BC_MY.value: "WaterWorldRGAndBCAndMY-v0",
    WaterWorldTasks.R_WO_G.value: "WaterWorldRWithoutG-v0",
    WaterWorldTasks.R_WO_GB.value: "WaterWorldRWithoutGB-v0",
    WaterWorldTasks.R_WO_GBC.value: "WaterWorldRWithoutGBC-v0",
    WaterWorldTasks.R_WO_GBCY.value: "WaterWorldRWithoutGBCY-v0",
    WaterWorldTasks.R_WO_GBCYM.value: "WaterWorldRWithoutGBCYM-v0",
    WaterWorldTasks.G_WO_B.value: "WaterWorldGWithoutB-v0",
    WaterWorldTasks.G_WO_BC.value: "WaterWorldGWithoutBC-v0",
    WaterWorldTasks.G_WO_BCY.value: "WaterWorldGWithoutBCY-v0",
    WaterWorldTasks.G_WO_BCYM.value: "WaterWorldGWithoutBCYM-v0",
    WaterWorldTasks.G_WO_BCYMR.value: "WaterWorldGWithoutBCYMR-v0",
    WaterWorldTasks.B_WO_C.value: "WaterWorldBWithoutC-v0",
    WaterWorldTasks.B_WO_CY.value: "WaterWorldBWithoutCY-v0",
    WaterWorldTasks.B_WO_CYM.value: "WaterWorldBWithoutCYM-v0",
    WaterWorldTasks.B_WO_CYMR.value: "WaterWorldBWithoutCYMR-v0",
    WaterWorldTasks.B_WO_CYMRG.value: "WaterWorldBWithoutCYMRG-v0",
    WaterWorldTasks.C_WO_Y.value: "WaterWorldCWithoutY-v0",
    WaterWorldTasks.C_WO_YM.value: "WaterWorldCWithoutYM-v0",
    WaterWorldTasks.C_WO_YMR.value: "WaterWorldCWithoutYMR-v0",
    WaterWorldTasks.C_WO_YMRG.value: "WaterWorldCWithoutYMRG-v0",
    WaterWorldTasks.C_WO_YMRGB.value: "WaterWorldCWithoutYMRGB-v0",
    WaterWorldTasks.Y_WO_M.value: "WaterWorldYWithoutM-v0",
    WaterWorldTasks.Y_WO_MR.value: "WaterWorldYWithoutMR-v0",
    WaterWorldTasks.Y_WO_MRG.value: "WaterWorldYWithoutMRG-v0",
    WaterWorldTasks.Y_WO_MRGB.value: "WaterWorldYWithoutMRGB-v0",
    WaterWorldTasks.Y_WO_MRGBC.value: "WaterWorldYWithoutMRGBC-v0",
    WaterWorldTasks.M_WO_R.value: "WaterWorldMWithoutR-v0",
    WaterWorldTasks.M_WO_RG.value: "WaterWorldMWithoutRG-v0",
    WaterWorldTasks.M_WO_RGB.value: "WaterWorldMWithoutRGB-v0",
    WaterWorldTasks.M_WO_RGBC.value: "WaterWorldMWithoutRGBC-v0",
    WaterWorldTasks.M_WO_RGBCY.value: "WaterWorldMWithoutRGBCY-v0",
    WaterWorldTasks.G_WO_BR.value: "WaterWorldGWithoutBR-v0",
    WaterWorldTasks.B_WO_R.value: "WaterWorldBWithoutR-v0",
    WaterWorldTasks.B_WO_RG.value: "WaterWorldBWithoutRG-v0",
    WaterWorldTasks.Y_WO_MC.value: "WaterWorldYWithoutMC-v0",
    WaterWorldTasks.M_WO_C.value: "WaterWorldMWithoutC-v0",
    WaterWorldTasks.M_WO_CY.value: "WaterWorldMWithoutCY-v0",
    WaterWorldTasks.RE_WO_G.value: "WaterWorldREWithoutG-v0",
    WaterWorldTasks.RE_WO_GB.value: "WaterWorldREWithoutGB-v0",
    WaterWorldTasks.GE_WO_B.value: "WaterWorldGEWithoutB-v0",
    WaterWorldTasks.GE_WO_BR.value: "WaterWorldGEWithoutBR-v0",
    WaterWorldTasks.BE_WO_R.value: "WaterWorldBEWithoutR-v0",
    WaterWorldTasks.BE_WO_RG.value: "WaterWorldBEWithoutRG-v0",
    WaterWorldTasks.CE_WO_M.value: "WaterWorldCEWithoutM-v0",
    WaterWorldTasks.CE_WO_MY.value: "WaterWorldCEWithoutMY-v0",
    WaterWorldTasks.ME_WO_Y.value: "WaterWorldMEWithoutY-v0",
    WaterWorldTasks.ME_WO_YC.value: "WaterWorldMEWithoutYC-v0",
    WaterWorldTasks.YE_WO_C.value: "WaterWorldYEWithoutC-v0",
    WaterWorldTasks.YE_WO_CM.value: "WaterWorldYEWithoutCM-v0",
    WaterWorldTasks.GE_WO_BC.value: "WaterWorldGEWithoutBC-v0",
    WaterWorldTasks.BE_WO_C.value: "WaterWorldBEWithoutC-v0",
    WaterWorldTasks.BE_WO_CM.value: "WaterWorldBEWithoutCM-v0",
    WaterWorldTasks.ME_WO_YR.value: "WaterWorldMEWithoutYR-v0",
    WaterWorldTasks.YE_WO_R.value: "WaterWorldYEWithoutR-v0",
    WaterWorldTasks.YE_WO_RG.value: "WaterWorldYEWithoutRG-v0",
    WaterWorldTasks.RGB_FULL_STRICT.value: "WaterWorldRGBFullStrict-v0",
    WaterWorldTasks.CMY_FULL_STRICT.value: "WaterWorldCMYFullStrict-v0",
    WaterWorldTasks.RGB_INTERAVOIDANCE.value: "WaterWorldRGBInteravoidance-v0",
    WaterWorldTasks.CMY_INTERAVOIDANCE.value: "WaterWorldCMYInteravoidance-v0",
    WaterWorldTasks.REGEBE_INTERAVOIDANCE.value: "WaterWorldRGBEmptyInteravoidance-v0",
    WaterWorldTasks.CEMEYE_INTERAVOIDANCE.value: "WaterWorldCMYEmptyInteravoidance-v0",
    WaterWorldTasks.REGEBE_AVOID_NEXT_TWO.value: "WaterWorldRGBEmptyAvoidNextTwo-v0",
    WaterWorldTasks.CEMEYE_AVOID_NEXT_TWO.value: "WaterWorldCMYEmptyAvoidNextTwo-v0",
    WaterWorldTasks.RGB_CMY_INTERAVOIDANCE.value: "WaterWorldRGBAndCMYInteravoidance-v0",
    WaterWorldTasks.REGEBE_CEMEYE_INTERAVOIDANCE.value: "WaterWorldRGBAndCYMEmptyInteravoidance-v0",
    WaterWorldTasks.REGEBE_CEMEYE_AVOID_NEXT_TWO.value: "WaterWorldRGBAndCMYEmptyAvoidNextTwo-v0"
}


def get_environment_class(env_name):
    if env_name in ENV_CLASSES:
        return f"{ENV_PREFIX}{ENV_CLASSES.get(env_name)}"
    raise RuntimeError(f"Error: The environment '{env_name}' does not exist.")


def get_random_tasks(params, env_name, num_tasks, use_environment_seed, starting_environment_seed):
    tasks = []
    env_class = get_environment_class(env_name)
    for task_id in range(num_tasks):
        if use_environment_seed:
            if starting_environment_seed is None:
                raise RuntimeError(f"Error: A starting seed must be specified for {env_name}.")
            seed = task_id + starting_environment_seed
        else:
            seed = None
        task_params = {**params, "generation": "random", "environment_seed": seed}
        tasks.append(gym.make(env_class, params=task_params))
    return tasks
