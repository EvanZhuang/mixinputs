import logging
import argparse
import os 

logger = logging.getLogger(__name__)

def setup_mixinputs(name, parser):
    subparser = parser.add_parser(
        name, 
        description="setup MixInputs", 
        help='setup MixInputs',
    )
    subparser.set_defaults(func=setup_mixinputs_runner)
    return subparser

def setup_mixinputs_runner(args):
    import site
    path = site.getsitepackages()[0]
    need_usercustomize = True
    if os.path.exists(os.path.join(path, 'usercustomize.py')):
        with open(os.path.join(path, 'usercustomize.py'), 'r') as f:
            for line in f.readlines():
                if 'import mixinputs' in line and not line.strip().startswith("#"):
                    logger.info("mixinputs already imported in usercustomize.py")
                    need_usercustomize = False
                    break 
                           
    if need_usercustomize:
        with open(os.path.join(path, 'usercustomize.py'), 'a') as f:
            f.write(f"try: import mixinputs\nexcept ImportError: pass\n")
            logger.info("mixinputs setup added to usercustomize.py")

    logger.info(f"MixInputs setup completed at location: {path}")
    logger.info(
        f"MixInputs Activated, set MIXINPUTS_BETA=<value> to enable it."
    )

def clean_up_mixinputs(name, parser):
    subparser = parser.add_parser(
        name,
        description="setup MixInputs",
        help='setup MixInputs',
    )
    subparser.set_defaults(func=cleanup_mixinputs_runner)
    return subparser

def cleanup_mixinputs_runner(args):
    import site
    path = site.getsitepackages()[0]
    if os.path.exists(os.path.join(path, 'usercustomize.py')):
        with open(os.path.join(path, 'usercustomize.py'), 'r') as f:
            lines = f.readlines()
            
        need_write = False 
        for i in range(len(lines)):
            if 'import mixinputs' in lines[i]:
                l_processed = lines[i].strip()
                if not l_processed.startswith("#"):
                    if l_processed == 'import mixinputs':
                        lines[i] = f"# {lines[i]}"
                        logger.info("mixinputs setup removed from usercustomize.py")
                        need_write = True
                    elif l_processed == 'try: import mixinputs':
                        lines[i] = f"# {lines[i]}"
                        lines[i+1] = f"# {lines[i+1]}"
                        logger.info("mixinputs setup removed from usercustomize.py")
                        need_write = True
                    else:
                        logger.warning(
                            "mixinputs setup found in usercustomize.py, "
                            "but not removed, due to unknown format."
                            f"please remove the command {lines[i]} manually."
                            f"from {os.path.join(path, 'usercustomize.py')}"
                        )
        
        if need_write:
            with open(os.path.join(path, 'usercustomize.py'), 'w') as f:
                f.writelines(lines)

def run():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            "setup": setup_mixinputs,
            "cleanup": clean_up_mixinputs,
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand(name, subparsers)

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
        