def Set_parameter(args):
    if args.env_name == "Swimmer-v2":
        args.pop_size = 10
        args.ada_init = 500
        args.RtoE = 3
        args.cov_alpha = 0.01
        args.gl_target = 0.03
    elif args.env_name == "HalfCheetah-v2":
        args.pop_size = 10
        args.ada_init = 400
        args.RtoE = 1
        args.cov_alpha = 0.03
        args.gl_target = 0.05
    elif args.env_name == "Hopper-v2":
        args.pop_size = 6
        args.ada_init = 400
        args.RtoE = 1
        args.cov_alpha = 0.03
        args.gl_target = 0.05
    elif args.env_name == "Walker2d-v2":
        args.pop_size = 6
        args.ada_init = 400
        args.RtoE = 1
        args.cov_alpha = 0.03
        args.gl_target = 0.05
    elif args.env_name == "Ant-v2":
        args.pop_size = 6
        args.ada_init = 400
        args.RtoE = 1
        args.cov_alpha = 0.03
        args.gl_target = 0.05
    else:
        print("Check the Environment")


    return args