# 提交作业

## Slurm 简介

!!! quote

    - [Quick Start User Guide](https://slurm.schedmd.com/quickstart.html)

Slurm 提供三大功能：

- 控制用户对节点的访问
- 提供并行任务执行框架
- 提供任务调度和队列

Slurm 中的几个实体：

- 节点（node）组成分区（partition），可以视为任务队列
- 任务（job）由一个或多个步骤（step）组成

## 提交任务

Slurm 有三种提交任务的方式：

- `srun`
- `sbatch`
- `salloc`

不论以哪种方式，我们都推荐使用脚本文件执行任务。脚本文件模板如下：

```shell title="job.sh"
#!/bin/bash
#SBATCH --job-name=hpl
#SBATCH --partition=M6
#SBATCH --nodes=4                # Number of nodes
#SBATCH --ntasks=8               # Number of MPI ranks
#SBATCH --ntasks-per-node=2      # Number of MPI ranks per node
#SBATCH --cpus-per-task=8        # Number of OpenMP threads for each MPI rank
#SBATCH --time=00:00:30
#SBATCH --mail-type=END,FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mail@zju.edu # Email address
#SBATCH --output=%x_%j.log     # Standard output and error log
## Command(s) to run (example):
module load openmpi hpl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun xhpl
```

输出文件名有一些特殊的格式：

| 格式 | 描述 |
| --- | --- |
| `%j` | 作业 ID |
| `%n` | 节点名<br>使用它可以将不同节点的输出分开 |
| `%x` | 任务名 |
| `%u` | 用户名 |

举例：

```text
job%2j-%2t.out
    job128-00.out, job128-01.out, ...
```

## 查看任务

要使用 `sacct` 命令查看历史任务，你可以按照以下步骤操作：

1. **基本使用**：直接使用 `sacct` 命令来查看当前或者过去任务的概况。你可以在终端中输入以下命令：

   ```bash
   sacct
   ```

   这将显示最近的任务信息，包括任务 ID、任务名称、状态等。

2. **指定时间范围**：如果你只想查看特定时间范围内的任务，可以使用 `--start` 和 `--end` 参数。例如，要查看从 2024 年 8 月 1 日到 8 月 31 日之间的任务，你可以使用：

   ```bash
   sacct --start=2024-08-01 --end=2024-08-31
   ```

3. **指定字段**：默认情况下，`sacct` 可能会显示许多你不关心的字段。你可以通过 `--format` 参数指定显示哪些字段。例如，如果你只想查看任务 ID、任务名称、状态和运行时间，可以使用：

   ```bash
   sacct --format=JobID,JobName,State,Elapsed
   ```

4. **查看某个特定任务**：如果你知道某个任务的 ID，并且想查看该任务的详细信息，可以直接使用任务 ID：

   ```bash
   sacct -j <JobID>
   ```

   例如：

   ```bash
   sacct -j 12345
   ```

5. **过滤状态**：你还可以使用 `-s` 参数来过滤任务状态。例如，查看所有失败的任务：

   ```bash
   sacct -s F
   ```

6. **更多帮助**：`sacct` 命令有许多选项，你可以使用 `man sacct` 或 `sacct --help` 查看所有可用选项及其详细说明。

通过这些步骤，你可以灵活地使用 `sacct` 命令查看并管理你的历史任务记录。
