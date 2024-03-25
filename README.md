def print_minefield(field):
    N = len(field) 

    result_field = [['0' for _ in range(N)] for _ in range(N)]

    for row in range(N):
        for col in range(N):

            if field[row][col] == 1:
                result_field[row][col] = '*'

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = row + dx, col + dy

                        if 0 <= nx < N and 0 <= ny < N and field[nx][ny] == 0:
                            result_field[nx][ny] = str(int(result_field[nx][ny]) + 1)


    for row in result_field:
        print(' '.join(row))



example_field = [
    [1, 0, 1, 1],
    [0, 0, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
]

print_minefield(example_field)