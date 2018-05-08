dependsOn(ProjectRef(file("DeepLearning.scala"), "plugins-Builtins"))

libraryDependencies += ("org.lwjgl" % "lwjgl" % "3.1.6" % Test).jar().classifier {
  import scala.util.Properties._
  if (isMac) {
    "natives-macos"
  } else if (isLinux) {
    "natives-linux"
  } else if (isWin) {
    "natives-windows"
  } else {
    throw new MessageOnlyException(s"lwjgl does not support $osName")
  }
}
