def disjoint_featuring_conversion_2_2(data, first_number_is_y=2, second_number_is_x=2):
    number_of_rows = int(28 / first_number_is_y)
    number_of_columns = int(28 / second_number_is_x)

    for identifier in range(0, len(data)):
        # declare a new array to transpose the original number into
        new_array = [0] * number_of_rows
        for i in range(0, number_of_rows):
            new_array[i] = ['0'] * number_of_columns
            # for j in range(0,number_of_columns):
            #    new_array[i][j]=[0]*second_number_is_x*first_number_is_y

        # now that new array is declared must fill it with data from the original
        og_x = 0
        og_y = 0
        for i in range(0, 14):
            for j in range(0, 14):
                new_array[i][j] = str(data[identifier][og_y][og_x]) + str(data[identifier][og_y][og_x + 1]) + str(
                    data[identifier][og_y + 1][og_x]) + str(data[identifier][og_y + 1][og_x + 1])
                # new_array[i][j][1] = data[identifier][og_y][og_x+1]
                # new_array[i][j][2] = data[identifier][og_y+1][og_x]
                # new_array[i][j][3] = data[identifier][og_y+1][og_x+1]
                og_x += 2
            og_y += 2
            og_x = 0
        data[identifier] = new_array


def disjoint_featuring_conversion_4_4(data, first_number_is_y=4, second_number_is_x=4):
    number_of_rows = int(28 / first_number_is_y)
    number_of_columns = int(28 / second_number_is_x)

    for identifier in range(0, len(data)):
        # declare a new array to transpose the original number into
        new_array = [0] * number_of_rows
        for i in range(0, number_of_rows):
            new_array[i] = [0] * number_of_columns
            # for j in range(0,number_of_columns):
            #    new_array[i][j]=[0]*second_number_is_x*first_number_is_y

        # now that new array is declared must fill it with data from the original
        og_x = 0
        og_y = 0
        for i in range(0, 7):
            for j in range(0, 7):
                new_array[i][j] = str(data[identifier][og_y][og_x]) + str(data[identifier][og_y][og_x + 1]) + str(
                    data[identifier][og_y][og_x + 2]) + str(data[identifier][og_y][og_x + 3]) + str(
                    data[identifier][og_y + 1][og_x]) + str(data[identifier][og_y + 1][og_x + 1]) + str(
                    data[identifier][og_y + 1][og_x + 2]) + str(data[identifier][og_y + 1][og_x + 3]) + str(
                    data[identifier][og_y + 2][og_x]) + str(data[identifier][og_y + 2][og_x + 1]) + str(
                    data[identifier][og_y + 2][og_x + 2]) + str(data[identifier][og_y + 2][og_x + 3]) + str(
                    data[identifier][og_y + 3][og_x]) + str(data[identifier][og_y + 3][og_x + 1]) + str(
                    data[identifier][og_y + 3][og_x + 2]) + str(data[identifier][og_y + 3][og_x + 3])

                og_x += 4
            og_y += 4
            og_x = 0
        data[identifier] = new_array


def disjoint_featuring_conversion_4_2(data, first_number_is_y=4, second_number_is_x=2):
    number_of_rows = int(28 / first_number_is_y)
    number_of_columns = int(28 / second_number_is_x)

    for identifier in range(0, len(data)):
        # declare a new array to transpose the original number into
        new_array = [0] * number_of_rows
        for i in range(0, number_of_rows):
            new_array[i] = [0] * number_of_columns
            # for j in range(0,number_of_columns):
            #    new_array[i][j]=[0]*second_number_is_x*first_number_is_y

        # now that new array is declared must fill it with data from the original
        og_x = 0
        og_y = 0
        for i in range(0, 7):
            for j in range(0, 14):
                new_array[i][j] = str(data[identifier][og_y][og_x]) + str(data[identifier][og_y][og_x + 1]) + str(
                    data[identifier][og_y + 1][og_x]) + str(data[identifier][og_y + 1][og_x + 1]) + str(
                    data[identifier][og_y + 2][og_x]) + str(data[identifier][og_y + 2][og_x + 1]) + str(
                    data[identifier][og_y + 3][og_x]) + str(data[identifier][og_y + 3][og_x + 1])
                og_x += 2
            og_y += 4
            og_x = 0
        data[identifier] = new_array


def disjoint_featuring_conversion_2_4(data, first_number_is_y=2, second_number_is_x=4):
    number_of_rows = int(28 / first_number_is_y)
    number_of_columns = int(28 / second_number_is_x)

    for identifier in range(0, len(data)):
        # declare a new array to transpose the original number into
        new_array = [0] * number_of_rows
        for i in range(0, number_of_rows):
            new_array[i] = [0] * number_of_columns
            # for j in range(0,number_of_columns):
            #    new_array[i][j]=[0]*second_number_is_x*first_number_is_y

        # now that new array is declared must fill it with data from the original
        og_x = 0
        og_y = 0
        for i in range(0, 14):
            for j in range(0, 7):
                new_array[i][j] = str(data[identifier][og_y][og_x]) + str(data[identifier][og_y][og_x + 1]) + str(
                    data[identifier][og_y][og_x + 2] + data[identifier][og_y][og_x + 3]) + str(
                    data[identifier][og_y + 1][og_x]) + str(data[identifier][og_y + 1][og_x + 1]) + str(
                    data[identifier][og_y + 1][og_x + 2] + data[identifier][og_y + 1][og_x + 3])
                og_x += 4
            og_y += 2
            og_x = 0
        data[identifier] = new_array
