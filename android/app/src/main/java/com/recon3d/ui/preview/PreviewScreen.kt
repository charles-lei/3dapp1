package com.recon3d.ui.preview

import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController

@Composable
fun PreviewScreen(nav: NavHostController, taskId: String) { Text("preview $taskId") }
