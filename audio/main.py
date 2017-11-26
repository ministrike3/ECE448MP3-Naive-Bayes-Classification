import mp21_22
import mp2_ec

if __name__=="__main__":

    # Make sure to run only one at a time for clarity, comment out the ones not in use

    # MP3 2.1 and 2.2 implementations

    #mp21_22.part_1()
    #mp21_22.part_2()

    # MP3 part 2 extra credit implementations

    #mp2_ec.part1()

    # To analyze LDA on digit data, yesno=False. To analyze on yesno corpus yesno=True
    #mp2_ec.part2(yesno=False)

    # To analyze on binarized data with bernoulli nb,  bin_val=True. To analyze with
    # probabiluty distribution with Multinomial nb, bin_val=False
    mp2_ec.part3(bin_val=True)
