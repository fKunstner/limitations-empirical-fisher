import efplt
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Experiment Runner')

    def add_options_to_group(options, group):
        for c in options:
            group.add_argument(c[0], action="store_true", help=c[1])

    experiment_choices = [
        ["-misspec", "Misspecification experiment"],
        ["-optim", "Optimization experiment"],
        ["-vecfield", "Vector field visualization"],
    ]

    add_options_to_group(experiment_choices, parser.add_argument_group('Experiment selection').add_mutually_exclusive_group(required=True))

    main_options = [
        ["-run", "Runs the experiment and save results as a .pk file"],
        ["-plot", "Plots the result from a .pk file (requires -save and/or -show)"],
        ["-appendix", "Also run/plot the experiments in the appendix"],
    ]
    add_options_to_group(main_options, parser.add_argument_group('Action selection', "At least one of [-run, -plot] is required"))

    plotting_options = [
        ["-save", "Save the plots"],
        ["-show", "Show the plots"],
    ]
    add_options_to_group(plotting_options, parser.add_argument_group('Plotting options', "At least one of [-save, -show] is required if plotting"))

    args = parser.parse_args()

    if not (args.run or args.plot):
        parser.error('No action requested, add -run and/or -plot')
    if args.plot and not (args.show or args.save):
        parser.error("-plot requires -save and/or -show.")

    return args


def savefigs(figs, expname):
    for i, fig in enumerate(figs):
        if isinstance(fig, list):
            for j, f in enumerate(fig):
                efplt.save(f, expname + "-" + str(i) + "-" + str(j) + ".pdf")
        else:
            efplt.save(fig, expname + "-" + str(i) + ".pdf")


if __name__ == "__main__":
    args = parse()
    print("")

    if args.vecfield:
        import vecfield.main as exp
        expname = "vecfield"
    if args.misspec:
        import misspec.main as exp
        expname = "misspec"
    if args.optim:
        import optim.main as exp
        expname = "optim"

    if args.run:
        if args.appendix:
            exp.run_appendix()
        else:
            exp.run()

    if args.plot:
        if args.appendix:
            figs = exp.plot_appendix()
        else:
            figs = exp.plot()

        if args.show:
            efplt.plt.show()

        if args.save:
            savefigs(figs, expname + ("-apx" if args.appendix else ""))
