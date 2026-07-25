package com.recon3d

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.recon3d.ui.capture.CaptureScreen
import com.recon3d.ui.home.HomeScreen
import com.recon3d.ui.preview.PreviewScreen
import com.recon3d.ui.settings.SettingsScreen
import com.recon3d.ui.task.TaskDetailScreen
import com.recon3d.ui.theme.ReconTheme

object Routes {
    const val HOME = "home"
    const val SETTINGS = "settings"
    const val CAPTURE = "capture"
    const val TASK = "task/{taskId}"
    const val PREVIEW = "preview/{taskId}"
    fun task(id: String) = "task/$id"
    fun preview(id: String) = "preview/$id"
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ReconTheme {
                val nav = rememberNavController()
                NavHost(navController = nav, startDestination = Routes.HOME) {
                    composable(Routes.HOME) { HomeScreen(nav) }
                    composable(Routes.SETTINGS) { SettingsScreen(nav) }
                    composable(Routes.CAPTURE) { CaptureScreen(nav) }
                    composable(Routes.TASK) { backStack ->
                        TaskDetailScreen(nav, backStack.arguments?.getString("taskId")!!)
                    }
                    composable(Routes.PREVIEW) { backStack ->
                        PreviewScreen(nav, backStack.arguments?.getString("taskId")!!)
                    }
                }
            }
        }
    }
}
