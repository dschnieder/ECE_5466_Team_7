package com.example.pokevision_mobile

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat.startForegroundService
import androidx.core.content.ContextCompat
import com.example.pokevision_mobile.ui.theme.Pokevision_mobileTheme

class MainActivity : ComponentActivity() {
    private val uiState = mutableStateOf(HunterUiState())

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                HunterContract.ACTION_STATE -> {
                    val connected = intent.getBooleanExtra(HunterContract.EXTRA_CONNECTED, false)
                    val status = intent.getStringExtra(HunterContract.EXTRA_STATUS).orEmpty()
                    val targets = intent.getStringArrayListExtra(HunterContract.EXTRA_TARGETS)?.toList().orEmpty()
                    uiState.value = uiState.value.copy(
                        connected = connected,
                        status = status,
                        activeTargets = targets,
                    )
                }

                HunterContract.ACTION_SHINY -> {
                    val label = intent.getStringExtra(HunterContract.EXTRA_LABEL).orEmpty()
                    val score = intent.getDoubleExtra(HunterContract.EXTRA_SCORE, 0.0)
                    val message = "$label (score ${"%.2f".format(score)})"
                    uiState.value = uiState.value.copy(
                        latestShinies = listOf(message) + uiState.value.latestShinies.take(19),
                    )
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        maybeRequestNotificationPermission()
        enableEdgeToEdge()

        val filter = IntentFilter().apply {
            addAction(HunterContract.ACTION_STATE)
            addAction(HunterContract.ACTION_SHINY)
        }
        ContextCompat.registerReceiver(this, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)

        setContent {
            Pokevision_mobileTheme {
                HunterScreen(uiState.value)
            }
        }
    }

    override fun onDestroy() {
        unregisterReceiver(receiver)
        super.onDestroy()
    }

    private fun maybeRequestNotificationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) {
            return
        }
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 42)
    }
}

@Composable
private fun HunterScreen(state: HunterUiState) {
    val context = LocalContext.current

    var serverUrl by remember { mutableStateOf(state.serverUrl) }
    var targetsInput by remember { mutableStateOf(state.targetsInput) }

    DisposableEffect(state.serverUrl) {
        serverUrl = state.serverUrl
        onDispose { }
    }
    DisposableEffect(state.targetsInput) {
        targetsInput = state.targetsInput
        onDispose { }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Pokevision Remote", style = MaterialTheme.typography.headlineSmall)
        Text(
            text = if (state.connected) "Connected" else "Disconnected",
            color = if (state.connected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
        )
        Text(text = state.status)

        OutlinedTextField(
            value = serverUrl,
            onValueChange = { serverUrl = it },
            label = { Text("Server URL") },
            placeholder = { Text("ws://192.168.1.10:8765") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
        )

        OutlinedTextField(
            value = targetsInput,
            onValueChange = { targetsInput = it },
            label = { Text("Target dex numbers") },
            placeholder = { Text("122, 25") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                startForegroundService(
                    context,
                    Intent(context, HunterForegroundService::class.java)
                        .setAction(HunterContract.ACTION_START)
                        .putExtra(HunterContract.EXTRA_SERVER_URL, serverUrl.trim())
                )
            }) {
                Text("Start")
            }

            Button(onClick = {
                context.startService(
                    Intent(context, HunterForegroundService::class.java)
                        .setAction(HunterContract.ACTION_STOP)
                )
            }) {
                Text("Stop")
            }

            Button(onClick = {
                val targets = parseTargets(targetsInput)
                context.startService(
                    Intent(context, HunterForegroundService::class.java)
                        .setAction(HunterContract.ACTION_SET_TARGETS)
                        .putStringArrayListExtra(HunterContract.EXTRA_TARGETS, ArrayList(targets))
                )
            }) {
                Text("Apply Targets")
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("Active targets", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(6.dp))
                val shownTargets = if (state.activeTargets.isEmpty()) "None" else state.activeTargets.joinToString(", ")
                Text(shownTargets)
            }
        }

        Text("Shiny alerts", style = MaterialTheme.typography.titleMedium)
        LazyColumn(modifier = Modifier.fillMaxWidth()) {
            items(state.latestShinies) { item ->
                Text(item, modifier = Modifier.padding(vertical = 2.dp))
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun HunterPreview() {
    Pokevision_mobileTheme {
        HunterScreen(HunterUiState(connected = true, status = "Connected", activeTargets = listOf("122"), latestShinies = listOf("122_shiny (score 0.95)")))
    }
}

private fun parseTargets(raw: String): List<String> = raw
    .split(",")
    .map { it.trim() }
    .filter { it.isNotEmpty() }
    .distinct()

private data class HunterUiState(
    val connected: Boolean = false,
    val status: String = "Idle",
    val serverUrl: String = "ws://192.168.1.10:8765",
    val targetsInput: String = "122",
    val activeTargets: List<String> = emptyList(),
    val latestShinies: List<String> = emptyList(),
)
