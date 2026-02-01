//! Silent Payments (BIP352) support for onchain transactions.
//!
//! This module provides functionality to send to Silent Payment addresses,
//! which allow receivers to have a static address that senders can use
//! to derive unique addresses for each payment, improving privacy.
//!
//! # Example
//!
//! ```no_run
//! use bark::onchain::silent_payment::is_silent_payment_address;
//! use bitcoin::{Amount, FeeRate};
//!
//! // Check if an address is a silent payment address
//! let is_sp = is_silent_payment_address("sp1qq...");
//!
//! // Send using the onchain wallet
//! // let txid = wallet.send_to_silent_payment(&chain, sp_address, amount, fee_rate).await?;
//! ```

use anyhow::Context;
use bitcoin::{Amount, FeeRate, Network, ScriptBuf, Txid, XOnlyPublicKey};
use bitcoin::key::TweakedPublicKey;
use log::{debug, info};

use silentpayments::sending::generate_recipient_pubkeys;
use silentpayments::utils::sending::calculate_partial_secret;
pub use silentpayments::{Network as SpNetwork, SilentPaymentAddress};
use silentpayments::secp256k1 as sp_secp;

use crate::chain::ChainSource;

#[cfg(feature = "onchain_bdk")]
use super::OnchainWallet;

#[cfg(feature = "onchain_bdk")]
use super::SignPsbt;

/// Error type for silent payment operations
#[derive(Debug, thiserror::Error)]
pub enum SilentPaymentError {
    /// The silent payment address could not be parsed.
    #[error("invalid silent payment address: {0}")]
    InvalidAddress(String),
    /// Network mismatch between wallet and silent payment address.
    #[error("network mismatch: wallet is on {wallet_network}, but address is for {address_network}")]
    NetworkMismatch {
        wallet_network: String,
        address_network: String,
    },
    /// Insufficient funds for the payment.
    #[error("insufficient funds: need {needed}, have {available}")]
    InsufficientFunds { needed: Amount, available: Amount },
    /// No eligible inputs for silent payment (need taproot or p2wpkh inputs).
    #[error("no eligible inputs for silent payment - need taproot or p2wpkh inputs")]
    NoEligibleInputs,
    /// Could not derive private keys for inputs.
    #[error("could not derive private keys for inputs: {0}")]
    KeyDerivationError(String),
    /// Error generating recipient public keys.
    #[error("error generating recipient pubkeys: {0}")]
    RecipientPubkeyError(String),
    /// Transaction building error.
    #[error("transaction building error: {0}")]
    TransactionError(String),
}

/// Convert bitcoin network to silentpayments network.
pub fn bitcoin_network_to_sp(network: Network) -> SpNetwork {
    match network {
        Network::Bitcoin => SpNetwork::Mainnet,
        Network::Testnet | Network::Signet | Network::Testnet4 => SpNetwork::Testnet,
        Network::Regtest => SpNetwork::Regtest,
    }
}

/// Check if a silent payment address network matches the wallet network.
pub fn check_network_match(
    sp_address: &SilentPaymentAddress,
    wallet_network: Network,
) -> Result<(), SilentPaymentError> {
    let expected_sp_network = bitcoin_network_to_sp(wallet_network);
    if sp_address.get_network() != expected_sp_network {
        return Err(SilentPaymentError::NetworkMismatch {
            wallet_network: format!("{:?}", wallet_network),
            address_network: format!("{:?}", sp_address.get_network()),
        });
    }
    Ok(())
}

/// Parse a silent payment address string.
///
/// Silent payment addresses start with:
/// - `sp1` for mainnet
/// - `tsp1` for testnet/signet  
/// - `sprt1` for regtest
pub fn parse_silent_payment_address(
    address: &str,
) -> Result<SilentPaymentAddress, SilentPaymentError> {
    SilentPaymentAddress::try_from(address)
        .map_err(|e| SilentPaymentError::InvalidAddress(format!("{:?}", e)))
}

/// Check if a string looks like a silent payment address.
///
/// This does a quick prefix check without full parsing.
pub fn is_silent_payment_address(address: &str) -> bool {
    address.starts_with("sp1")
        || address.starts_with("tsp1")
        || address.starts_with("sprt1")
}

/// Convert a bitcoin secp256k1 secret key to silentpayments secp256k1 secret key.
/// 
/// This is needed because the two crates use different versions of secp256k1.
fn convert_secret_key(key: &bitcoin::secp256k1::SecretKey) -> sp_secp::SecretKey {
    sp_secp::SecretKey::from_slice(&key.secret_bytes())
        .expect("valid secret key bytes")
}

/// Convert a silentpayments XOnlyPublicKey to a bitcoin TweakedPublicKey for P2TR.
///
/// Note: This assumes the key is already the final output key (properly tweaked).
fn convert_xonly_to_tweaked(key: &sp_secp::XOnlyPublicKey) -> TweakedPublicKey {
    let bytes = key.serialize();
    let xonly = XOnlyPublicKey::from_slice(&bytes).expect("valid xonly pubkey bytes");
    TweakedPublicKey::dangerous_assume_tweaked(xonly)
}

/// Derive the output script for a silent payment.
///
/// Given a silent payment address and the input private keys with their outpoints,
/// this derives the unique taproot output script that should be used for the payment.
///
/// # Arguments
///
/// * `sp_address` - The silent payment address to send to
/// * `input_keys` - A slice of (bitcoin SecretKey, is_taproot) tuples for each input
/// * `outpoints` - The outpoints being spent, as (txid_hex_string, vout) tuples
///
/// # Returns
///
/// The derived taproot script pubkey for the output.
pub fn derive_silent_payment_script(
    sp_address: &SilentPaymentAddress,
    input_keys: &[(bitcoin::secp256k1::SecretKey, bool)],
    outpoints: &[(String, u32)],
) -> Result<ScriptBuf, SilentPaymentError> {
    // Convert the input keys to silentpayments secp256k1 version
    let sp_input_keys: Vec<(sp_secp::SecretKey, bool)> = input_keys
        .iter()
        .map(|(key, is_taproot)| (convert_secret_key(key), *is_taproot))
        .collect();

    // Calculate the partial secret from the input keys
    let partial_secret = calculate_partial_secret(&sp_input_keys, outpoints)
        .map_err(|e| SilentPaymentError::RecipientPubkeyError(format!("{:?}", e)))?;

    // Generate the recipient pubkeys
    let outputs = generate_recipient_pubkeys(vec![sp_address.clone()], partial_secret)
        .map_err(|e| SilentPaymentError::RecipientPubkeyError(format!("{:?}", e)))?;

    // Get the first (and only) output pubkey for this address
    let pubkeys = outputs
        .get(sp_address)
        .ok_or_else(|| SilentPaymentError::RecipientPubkeyError(
            "no output generated for address".to_string()
        ))?;

    let output_pubkey = pubkeys
        .first()
        .ok_or_else(|| SilentPaymentError::RecipientPubkeyError(
            "empty pubkey list".to_string()
        ))?;

    // Convert the silentpayments XOnlyPublicKey to a bitcoin TweakedPublicKey
    let tweaked_pubkey = convert_xonly_to_tweaked(output_pubkey);

    // Create a P2TR script for the output
    Ok(ScriptBuf::new_p2tr_tweaked(tweaked_pubkey))
}

/// Represents a prepared silent payment with the derived output information.
#[derive(Debug, Clone)]
pub struct PreparedSilentPayment {
    /// The original silent payment address.
    pub address: SilentPaymentAddress,
    /// The derived output x-only public key.
    pub output_pubkey: XOnlyPublicKey,
    /// The derived taproot script pubkey.
    pub script_pubkey: ScriptBuf,
    /// The amount to send.
    pub amount: Amount,
}

#[cfg(feature = "onchain_bdk")]
impl OnchainWallet {
    /// Send to a silent payment address.
    ///
    /// This method:
    /// 1. Selects inputs from the wallet (taproot preferred)
    /// 2. Derives the unique output address using BIP352
    /// 3. Creates and broadcasts the transaction
    ///
    /// # Arguments
    ///
    /// * `chain` - The chain source for broadcasting
    /// * `sp_address` - The silent payment address to send to
    /// * `amount` - The amount to send
    /// * `fee_rate` - The fee rate to use
    ///
    /// # Returns
    ///
    /// The txid of the broadcast transaction.
    pub async fn send_to_silent_payment(
        &mut self,
        chain: &ChainSource,
        sp_address: SilentPaymentAddress,
        amount: Amount,
        fee_rate: FeeRate,
    ) -> anyhow::Result<Txid> {
        use bdk_wallet::KeychainKind;
        use bitcoin::secp256k1::Secp256k1;

        let network = self.inner.network();

        check_network_match(&sp_address, network)
            .context("network check failed")?;

        info!("Preparing silent payment to {} for {}", sp_address, amount);

        let mut builder = self.inner.build_tx();
        
        let placeholder_bytes = [0x02u8; 32];
        let placeholder_xonly = XOnlyPublicKey::from_slice(&placeholder_bytes)
            .expect("valid placeholder");
        let temp_script = ScriptBuf::new_p2tr_tweaked(
            TweakedPublicKey::dangerous_assume_tweaked(placeholder_xonly)
        );
        builder.add_recipient(temp_script.clone(), amount);
        builder.fee_rate(fee_rate);

        let temp_psbt = builder.finish()
            .context("error building temporary transaction")?;

        let mut input_keys: Vec<(bitcoin::secp256k1::SecretKey, bool)> = Vec::new();
        let mut outpoints: Vec<(String, u32)> = Vec::new();

        let secp = Secp256k1::new();

        for (idx, input) in temp_psbt.unsigned_tx.input.iter().enumerate() {
            let outpoint = input.previous_output;
            outpoints.push((outpoint.txid.to_string(), outpoint.vout));

            let utxo = self.inner.get_utxo(outpoint)
                .context(format!("could not find UTXO for input {}", idx))?;
            
            let is_taproot = utxo.txout.script_pubkey.is_p2tr();
            
            let (_keychain, derivation_index) = self.inner.derivation_of_spk(utxo.txout.script_pubkey.clone())
                .context(format!("could not find derivation for input {} script", idx))?;
            
            let signers = self.inner.get_signers(KeychainKind::External);
            
            #[allow(deprecated)]
            let secret_key = signers.signers()
                .iter()
                .filter_map(|s| s.descriptor_secret_key())
                .find_map(|dsk| {
                    // Get the xpriv and derive the key at the specific index
                    match dsk {
                        bdk_wallet::keys::DescriptorSecretKey::XPrv(xprv_desc) => {
                            // Derive the child key at the given index
                            let path = bitcoin::bip32::DerivationPath::from(vec![
                                bitcoin::bip32::ChildNumber::Normal { index: derivation_index }
                            ]);
                            let derived = xprv_desc.xkey.derive_priv(&secp, &path).ok()?;
                            Some(derived.private_key)
                        },
                        bdk_wallet::keys::DescriptorSecretKey::Single(single) => {
                            Some(single.key.inner)
                        },
                        _ => None,
                    }
                })
                .context(format!("could not derive secret key for input {}", idx))?;

            input_keys.push((secret_key, is_taproot));
        }

        if input_keys.is_empty() {
            return Err(SilentPaymentError::NoEligibleInputs.into());
        }

        debug!("Using {} inputs for silent payment derivation", input_keys.len());

        let output_script = derive_silent_payment_script(&sp_address, &input_keys, &outpoints)
            .context("error deriving silent payment script")?;

        info!("Derived silent payment output script: {}", output_script.to_hex_string());

        let mut final_builder = self.inner.build_tx();
        final_builder.add_recipient(output_script, amount);
        final_builder.fee_rate(fee_rate);

        for outpoint in temp_psbt.unsigned_tx.input.iter().map(|i| i.previous_output) {
            final_builder.add_utxo(outpoint)
                .context("error adding utxo to final tx")?;
        }

        let psbt = final_builder.finish()
            .context("error building final transaction")?;

        let tx: bitcoin::Transaction = self.finish_tx(psbt).await?;
        let txid = tx.compute_txid();

        chain.broadcast_tx(&tx).await
            .context("error broadcasting transaction")?;

        info!("Silent payment transaction broadcast: {}", txid);

        Ok(txid)
    }
}
