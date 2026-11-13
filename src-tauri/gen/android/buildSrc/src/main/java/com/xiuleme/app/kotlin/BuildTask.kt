import java.io.File
import org.apache.tools.ant.taskdefs.condition.Os
import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.logging.LogLevel
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.TaskAction

open class BuildTask : DefaultTask() {
    @Input
    var rootDirRel: String? = null
    @Input
    var target: String? = null
    @Input
    var release: Boolean? = null

    @TaskAction
    fun assemble() {
        val executable = """npm""";
        try {
            runTauriCli(executable)
        } catch (e: Exception) {
            if (Os.isFamily(Os.FAMILY_WINDOWS)) {
                // Prefer absolute npm.cmd via cmd.exe — Gradle CreateProcess often
                // cannot launch bare .cmd/.bat, and broken PATH entries break lookup.
                val absoluteNpm = findWindowsNpmCmd()
                if (absoluteNpm != null) {
                    try {
                        runTauriCliViaCmd(absoluteNpm)
                        return
                    } catch (absoluteException: Exception) {
                        // fall through to extension fallbacks
                    }
                }

                val fallbacks = listOf(
                    "$executable.exe",
                    "$executable.cmd",
                    "$executable.bat",
                )

                var lastException: Exception = e
                for (fallback in fallbacks) {
                    try {
                        runTauriCliViaCmd(fallback)
                        return
                    } catch (fallbackException: Exception) {
                        lastException = fallbackException
                    }
                }
                throw lastException
            } else {
                throw e;
            }
        }
    }

    private fun findWindowsNpmCmd(): String? {
        val candidates = mutableListOf<File>()
        System.getenv("PATH")?.split(File.pathSeparatorChar)?.forEach { raw ->
            val dir = raw.trim()
            // Skip empty / relative junk from corrupted PATH (e.g. "ram Files\nodejs\")
            if (dir.isEmpty() || !dir.contains(':')) return@forEach
            candidates.add(File(dir, "npm.cmd"))
        }
        candidates.add(File("""C:\nvm4w\nodejs\npm.cmd"""))
        candidates.add(File("""C:\Program Files\nodejs\npm.cmd"""))
        candidates.add(File("""D:\Program Files\nodejs\npm.cmd"""))
        return candidates.firstOrNull { it.isFile }?.absolutePath
    }

    fun runTauriCli(executable: String) {
        val rootDirRel = rootDirRel ?: throw GradleException("rootDirRel cannot be null")
        val target = target ?: throw GradleException("target cannot be null")
        val release = release ?: throw GradleException("release cannot be null")
        val args = listOf("run", "--", "tauri", "android", "android-studio-script");

        project.exec {
            workingDir(File(project.projectDir, rootDirRel))
            executable(executable)
            args(args)
            if (project.logger.isEnabled(LogLevel.DEBUG)) {
                args("-vv")
            } else if (project.logger.isEnabled(LogLevel.INFO)) {
                args("-v")
            }
            if (release) {
                args("--release")
            }
            args(listOf("--target", target))
        }.assertNormalExitValue()
    }

    private fun runTauriCliViaCmd(npmExecutable: String) {
        val rootDirRel = rootDirRel ?: throw GradleException("rootDirRel cannot be null")
        val target = target ?: throw GradleException("target cannot be null")
        val release = release ?: throw GradleException("release cannot be null")

        val cmdArgs = mutableListOf(
            "/c",
            npmExecutable,
            "run",
            "--",
            "tauri",
            "android",
            "android-studio-script",
        )
        if (project.logger.isEnabled(LogLevel.DEBUG)) {
            cmdArgs.add("-vv")
        } else if (project.logger.isEnabled(LogLevel.INFO)) {
            cmdArgs.add("-v")
        }
        if (release) {
            cmdArgs.add("--release")
        }
        cmdArgs.add("--target")
        cmdArgs.add(target)

        project.exec {
            workingDir(File(project.projectDir, rootDirRel))
            executable("cmd.exe")
            args(cmdArgs)
        }.assertNormalExitValue()
    }
}
