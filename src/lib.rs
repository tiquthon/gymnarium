//! # Gymnarium
//!
//! `gymnarium` contains all crates from the gymnarium framework.
//! For a quick start and most other use cases it's sufficient to include this as a single
//! dependency.
//!
//! Furthermore there are various run functions incorporating necessary elements in a loop.

extern crate bincode;
// pub extern crate gymnarium_agents_q_learning;
pub extern crate gymnarium_agents_random;
pub extern crate gymnarium_base;
pub extern crate gymnarium_environments_gym;
pub extern crate gymnarium_environments_tiquthon;
pub extern crate gymnarium_visualisers_base;
pub extern crate gymnarium_visualisers_piston;
extern crate ron;
extern crate serde;
extern crate serde_json;

use std::error::Error;
use std::fmt::{Debug, Display};
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Serialize;

use gymnarium_base::{Agent, Environment, Reward, Seed};

use gymnarium_visualisers_base::{
    DrawableEnvironment, PixelArrayDrawableEnvironment, PixelArrayVisualiser,
    TextDrawableEnvironment, TextVisualiser, ThreeDimensionalDrawableEnvironment,
    ThreeDimensionalVisualiser, TwoDimensionalDrawableEnvironment, TwoDimensionalVisualiser,
    Visualiser,
};

/* -- -- -- -- -- -- -- -- -- -- -- -- - FURTHER STRUCTURES - -- -- -- -- -- -- -- -- -- -- -- -- */

pub struct RunOptions {
    pub seed: Option<Seed>,
    pub reset_environment_on_done: bool,
    pub reset_agent_on_done: bool,
    pub environment_load_path: Option<String>,
    pub environment_store_path: Option<String>,
    pub agent_load_path: Option<String>,
    pub agent_store_path: Option<String>,
}

/* -- -- -- -- -- -- -- -- -- -- -- -- --  NO VISUALISER   -- -- -- -- -- -- -- -- -- -- -- -- -- */

pub fn run_with_no_visualiser<
    EError: Error,
    EReward: Reward,
    EInfo: Debug,
    EData: Serialize + DeserializeOwned,
    E: Environment<EError, EReward, EInfo, EData>,
    AError: Error,
    AData: Serialize + DeserializeOwned,
    A: Agent<AError, EReward, AData>,
    XCF: Fn(&E, &A, u128, u128) -> bool,
>(
    environment: E,
    agent: A,
    exit_condition: XCF,
    run_options: RunOptions,
) {
    let mut environment = environment;
    let mut agent = agent;

    let mut state = if let Some(environment_load_path_string) = run_options.environment_load_path {
        load_environment(&mut environment, environment_load_path_string).unwrap();
        environment.state()
    } else {
        environment.reseed(run_options.seed.clone()).unwrap();
        environment.reset().unwrap()
    };

    if let Some(agent_load_path_string) = run_options.agent_load_path {
        load_agent(&mut agent, agent_load_path_string).unwrap();
    } else {
        agent.reseed(run_options.seed).unwrap();
        agent.reset().unwrap();
    }

    let mut episode = 0u128;
    let mut step = 0u128;

    while !exit_condition(&environment, &agent, episode, step) {
        let action = agent.choose_action(&state).unwrap();

        let (new_state, reward, done, _) = environment.step(&action).unwrap();
        step += 1;
        agent
            .process_reward(&state, &action, &new_state, reward, done)
            .unwrap();

        state = if run_options.reset_environment_on_done && done {
            step = 0;
            episode += 1;
            environment.reset().unwrap()
        } else {
            new_state
        };

        if run_options.reset_agent_on_done && done {
            agent.reset().unwrap();
        }
    }

    if let Some(agent_store_path_string) = run_options.agent_store_path {
        store_agent(&agent, agent_store_path_string).unwrap();
    }

    if let Some(environment_store_path_string) = run_options.environment_store_path {
        store_environment(&environment, environment_store_path_string).unwrap();
    }

    agent.close().unwrap();
    environment.close().unwrap();
}

/* -- -- -- -- -- -- -- -- -- -- --  TWO DIMENSIONAL VISUALISER  -- -- -- -- -- -- -- -- -- -- -- */

pub fn run_with_two_dimensional_visualiser<
    EError: Error,
    EReward: Reward,
    EInfo: Debug,
    DEError: Error,
    EData: Serialize + DeserializeOwned,
    E: Environment<EError, EReward, EInfo, EData>
        + DrawableEnvironment
        + TwoDimensionalDrawableEnvironment<DEError>,
    AError: Error,
    AData: Serialize + DeserializeOwned,
    A: Agent<AError, EReward, AData>,
    VError: Error,
    TDVError: Error,
    V: Visualiser<VError> + TwoDimensionalVisualiser<TDVError, VError, DEError>,
    XCF: Fn(&E, &A, &V, u128, u128) -> bool,
>(
    environment: E,
    agent: A,
    visualiser: V,
    exit_condition: XCF,
    run_options: RunOptions,
) {
    let mut environment = environment;
    let mut agent = agent;
    let mut visualiser = visualiser;

    let mut state = if let Some(environment_load_path_string) = run_options.environment_load_path {
        load_environment(&mut environment, environment_load_path_string).unwrap();
        environment.state()
    } else {
        environment.reseed(run_options.seed.clone()).unwrap();
        environment.reset().unwrap()
    };

    visualiser.render_two_dimensional(&environment).unwrap();

    if let Some(agent_load_path_string) = run_options.agent_load_path {
        load_agent(&mut agent, agent_load_path_string).unwrap();
    } else {
        agent.reseed(run_options.seed).unwrap();
        agent.reset().unwrap();
    }

    let mut episode = 0u128;
    let mut step = 0u128;

    while !exit_condition(&environment, &agent, &visualiser, episode, step) {
        let action = agent.choose_action(&state).unwrap();

        let (new_state, reward, done, _) = environment.step(&action).unwrap();
        step += 1;

        agent
            .process_reward(&state, &action, &new_state, reward, done)
            .unwrap();

        state = if run_options.reset_environment_on_done && done {
            step = 0;
            episode += 1;
            environment.reset().unwrap()
        } else {
            new_state
        };

        if run_options.reset_agent_on_done && done {
            agent.reset().unwrap();
        }

        visualiser.render_two_dimensional(&environment).unwrap();

        sleep_suggested_steps_per_second_or_30_fps::<E>();
    }

    if let Some(agent_store_path_string) = run_options.agent_store_path {
        store_agent(&agent, agent_store_path_string).unwrap();
    }

    if let Some(environment_store_path_string) = run_options.environment_store_path {
        store_environment(&environment, environment_store_path_string).unwrap();
    }

    agent.close().unwrap();
    environment.close().unwrap();
    visualiser.close().unwrap();
}

/* -- -- -- -- -- -- -- -- -- -- -- THREE DIMENSIONAL VISUALISER -- -- -- -- -- -- -- -- -- -- -- */

pub fn _run_with_three_dimensional_visualiser<
    EError: Error,
    EReward: Reward,
    EInfo: Debug,
    DEError: Error,
    EData: Serialize + DeserializeOwned,
    E: Environment<EError, EReward, EInfo, EData>
        + DrawableEnvironment
        + ThreeDimensionalDrawableEnvironment<DEError>,
    AError: Error,
    AData: Serialize + DeserializeOwned,
    A: Agent<AError, EReward, AData>,
    VError: Error,
    TDVError: Error,
    V: Visualiser<VError> + ThreeDimensionalVisualiser<TDVError, VError, DEError>,
    XCF: Fn(&E, &A, &V, u128, u128) -> bool,
>(
    environment: E,
    agent: A,
    visualiser: V,
    exit_condition: XCF,
    run_options: RunOptions,
) {
    let mut environment = environment;
    let mut agent = agent;
    let mut visualiser = visualiser;

    let mut state = if let Some(environment_load_path_string) = run_options.environment_load_path {
        load_environment(&mut environment, environment_load_path_string).unwrap();
        environment.state()
    } else {
        environment.reseed(run_options.seed.clone()).unwrap();
        environment.reset().unwrap()
    };

    visualiser.render_three_dimensional(&environment).unwrap();

    if let Some(agent_load_path_string) = run_options.agent_load_path {
        load_agent(&mut agent, agent_load_path_string).unwrap();
    } else {
        agent.reseed(run_options.seed).unwrap();
        agent.reset().unwrap();
    }

    let mut episode = 0u128;
    let mut step = 0u128;

    while !exit_condition(&environment, &agent, &visualiser, episode, step) {
        let action = agent.choose_action(&state).unwrap();

        let (new_state, reward, done, _) = environment.step(&action).unwrap();
        step += 1;

        agent
            .process_reward(&state, &action, &new_state, reward, done)
            .unwrap();

        state = if run_options.reset_environment_on_done && done {
            step = 0;
            episode += 1;
            environment.reset().unwrap()
        } else {
            new_state
        };

        if run_options.reset_agent_on_done && done {
            agent.reset().unwrap();
        }

        visualiser.render_three_dimensional(&environment).unwrap();

        sleep_suggested_steps_per_second_or_30_fps::<E>();
    }

    if let Some(agent_store_path_string) = run_options.agent_store_path {
        store_agent(&agent, agent_store_path_string).unwrap();
    }

    if let Some(environment_store_path_string) = run_options.environment_store_path {
        store_environment(&environment, environment_store_path_string).unwrap();
    }

    agent.close().unwrap();
    environment.close().unwrap();
    visualiser.close().unwrap();
}

/* -- -- -- -- -- -- -- -- -- -- -- -- PIXEL ARRAY VISUALISER -- -- -- -- -- -- -- -- -- -- -- -- */

pub fn _run_with_pixel_array_visualiser<
    EError: Error,
    EReward: Reward,
    EInfo: Debug,
    DEError: Error,
    EData: Serialize + DeserializeOwned,
    E: Environment<EError, EReward, EInfo, EData>
        + DrawableEnvironment
        + PixelArrayDrawableEnvironment<DEError>,
    AError: Error,
    AData: Serialize + DeserializeOwned,
    A: Agent<AError, EReward, AData>,
    VError: Error,
    TDVError: Error,
    V: Visualiser<VError> + PixelArrayVisualiser<TDVError, VError, DEError>,
    XCF: Fn(&E, &A, &V, u128, u128) -> bool,
>(
    environment: E,
    agent: A,
    visualiser: V,
    exit_condition: XCF,
    run_options: RunOptions,
) {
    let mut environment = environment;
    let mut agent = agent;
    let mut visualiser = visualiser;

    let mut state = if let Some(environment_load_path_string) = run_options.environment_load_path {
        load_environment(&mut environment, environment_load_path_string).unwrap();
        environment.state()
    } else {
        environment.reseed(run_options.seed.clone()).unwrap();
        environment.reset().unwrap()
    };

    visualiser.render_pixel_array(&environment).unwrap();

    if let Some(agent_load_path_string) = run_options.agent_load_path {
        load_agent(&mut agent, agent_load_path_string).unwrap();
    } else {
        agent.reseed(run_options.seed).unwrap();
        agent.reset().unwrap();
    }

    let mut episode = 0u128;
    let mut step = 0u128;

    while !exit_condition(&environment, &agent, &visualiser, episode, step) {
        let action = agent.choose_action(&state).unwrap();

        let (new_state, reward, done, _) = environment.step(&action).unwrap();
        step += 1;

        agent
            .process_reward(&state, &action, &new_state, reward, done)
            .unwrap();

        state = if run_options.reset_environment_on_done && done {
            step = 0;
            episode += 1;
            environment.reset().unwrap()
        } else {
            new_state
        };

        if run_options.reset_agent_on_done && done {
            agent.reset().unwrap();
        }

        visualiser.render_pixel_array(&environment).unwrap();

        sleep_suggested_steps_per_second_or_30_fps::<E>();
    }

    if let Some(agent_store_path_string) = run_options.agent_store_path {
        store_agent(&agent, agent_store_path_string).unwrap();
    }

    if let Some(environment_store_path_string) = run_options.environment_store_path {
        store_environment(&environment, environment_store_path_string).unwrap();
    }

    agent.close().unwrap();
    environment.close().unwrap();
    visualiser.close().unwrap();
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- TEXT VISUALISER  -- -- -- -- -- -- -- -- -- -- -- -- -- */

pub fn _run_with_text_visualiser<
    EError: Error,
    EReward: Reward,
    EInfo: Debug,
    DEError: Error,
    EData: Serialize + DeserializeOwned,
    E: Environment<EError, EReward, EInfo, EData>
        + DrawableEnvironment
        + TextDrawableEnvironment<DEError>,
    AError: Error,
    AData: Serialize + DeserializeOwned,
    A: Agent<AError, EReward, AData>,
    VError: Error,
    TDVError: Error,
    V: Visualiser<VError> + TextVisualiser<TDVError, VError, DEError>,
    XCF: Fn(&E, &A, &V, u128, u128) -> bool,
>(
    environment: E,
    agent: A,
    visualiser: V,
    exit_condition: XCF,
    run_options: RunOptions,
) {
    let mut environment = environment;
    let mut agent = agent;
    let mut visualiser = visualiser;

    let mut state = if let Some(environment_load_path_string) = run_options.environment_load_path {
        load_environment(&mut environment, environment_load_path_string).unwrap();
        environment.state()
    } else {
        environment.reseed(run_options.seed.clone()).unwrap();
        environment.reset().unwrap()
    };

    visualiser.render_text(&environment).unwrap();

    if let Some(agent_load_path_string) = run_options.agent_load_path {
        load_agent(&mut agent, agent_load_path_string).unwrap();
    } else {
        agent.reseed(run_options.seed).unwrap();
        agent.reset().unwrap();
    }

    let mut episode = 0u128;
    let mut step = 0u128;

    while !exit_condition(&environment, &agent, &visualiser, episode, step) {
        let action = agent.choose_action(&state).unwrap();

        let (new_state, reward, done, _) = environment.step(&action).unwrap();
        step += 1;

        agent
            .process_reward(&state, &action, &new_state, reward, done)
            .unwrap();

        state = if run_options.reset_environment_on_done && done {
            step = 0;
            episode += 1;
            environment.reset().unwrap()
        } else {
            new_state
        };

        if run_options.reset_agent_on_done && done {
            agent.reset().unwrap();
        }

        visualiser.render_text(&environment).unwrap();

        sleep_suggested_steps_per_second_or_30_fps::<E>();
    }

    if let Some(agent_store_path_string) = run_options.agent_store_path {
        store_agent(&agent, agent_store_path_string).unwrap();
    }

    if let Some(environment_store_path_string) = run_options.environment_store_path {
        store_environment(&environment, environment_store_path_string).unwrap();
    }

    agent.close().unwrap();
    environment.close().unwrap();
    visualiser.close().unwrap();
}

/* -- -- -- -- -- -- -- -- -- -- -- EXIT CONDITION CONSTRUCTORS  -- -- -- -- -- -- -- -- -- -- -- */

/*
 * Visualiser:
 * - No Visualiser
 * - Two Dimensional Visualiser
 * - Three Dimensional Visualiser
 * - RGB Visualiser
 * - Text Visualiser
 *
 * Exit Variant:
 * - True
 * - False
 * - Episodes Simulated
 * - Visualiser Closed
 */

pub mod exit_condition {
    use super::*;

    pub mod when_no_visualiser {
        use super::*;

        pub fn as_soon_as_possible<
            EError: Error,
            EReward: Reward,
            EInfo: Debug,
            EData: Serialize + DeserializeOwned,
            E: Environment<EError, EReward, EInfo, EData>,
            AError: Error,
            AData: Serialize + DeserializeOwned,
            A: Agent<AError, EReward, AData>,
        >() -> impl Fn(&E, &A, u128, u128) -> bool {
            |_environment, _agent, _episode, _step| true
        }

        pub fn never<
            EError: Error,
            EReward: Reward,
            EInfo: Debug,
            EData: Serialize + DeserializeOwned,
            E: Environment<EError, EReward, EInfo, EData>,
            AError: Error,
            AData: Serialize + DeserializeOwned,
            A: Agent<AError, EReward, AData>,
        >() -> impl Fn(&E, &A, u128, u128) -> bool {
            |_environment, _agent, _episode, _step| false
        }

        pub fn episodes_simulated<
            EError: Error,
            EReward: Reward,
            EInfo: Debug,
            EData: Serialize + DeserializeOwned,
            E: Environment<EError, EReward, EInfo, EData>,
            AError: Error,
            AData: Serialize + DeserializeOwned,
            A: Agent<AError, EReward, AData>,
        >(
            count_of_episodes: u128,
        ) -> impl Fn(&E, &A, u128, u128) -> bool {
            move |_environment, _agent, episode, _step| episode >= count_of_episodes
        }
    }

    pub mod when_visualiser {
        use super::*;

        pub fn as_soon_as_possible<
            EError: Error,
            EReward: Reward,
            EInfo: Debug,
            EData: Serialize + DeserializeOwned,
            E: Environment<EError, EReward, EInfo, EData> + DrawableEnvironment,
            AError: Error,
            AData: Serialize + DeserializeOwned,
            A: Agent<AError, EReward, AData>,
            VError: Error,
            V: Visualiser<VError>,
        >() -> impl Fn(&E, &A, &V, u128, u128) -> bool {
            |_environment, _agent, _visualiser, _episode, _step| true
        }

        pub fn never<
            EError: Error,
            EReward: Reward,
            EInfo: Debug,
            EData: Serialize + DeserializeOwned,
            E: Environment<EError, EReward, EInfo, EData> + DrawableEnvironment,
            AError: Error,
            AData: Serialize + DeserializeOwned,
            A: Agent<AError, EReward, AData>,
            VError: Error,
            V: Visualiser<VError>,
        >() -> impl Fn(&E, &A, &V, u128, u128) -> bool {
            |_environment, _agent, _visualiser, _episode, _step| false
        }

        pub fn episodes_simulated<
            EError: Error,
            EReward: Reward,
            EInfo: Debug,
            EData: Serialize + DeserializeOwned,
            E: Environment<EError, EReward, EInfo, EData> + DrawableEnvironment,
            AError: Error,
            AData: Serialize + DeserializeOwned,
            A: Agent<AError, EReward, AData>,
            VError: Error,
            V: Visualiser<VError>,
        >(
            count_of_episodes: u128,
        ) -> impl Fn(&E, &A, &V, u128, u128) -> bool {
            move |_environment, _agent, _visualiser, episode, _step| episode >= count_of_episodes
        }

        pub fn closed<
            EError: Error,
            EReward: Reward,
            EInfo: Debug,
            EData: Serialize + DeserializeOwned,
            E: Environment<EError, EReward, EInfo, EData> + DrawableEnvironment,
            AError: Error,
            AData: Serialize + DeserializeOwned,
            A: Agent<AError, EReward, AData>,
            VError: Error,
            V: Visualiser<VError>,
        >() -> impl Fn(&E, &A, &V, u128, u128) -> bool {
            |_environment, _agent, visualiser, _episode, _step| !visualiser.is_open()
        }

        pub fn closed_or_episodes_simulated<
            EError: Error,
            EReward: Reward,
            EInfo: Debug,
            EData: Serialize + DeserializeOwned,
            E: Environment<EError, EReward, EInfo, EData> + DrawableEnvironment,
            AError: Error,
            AData: Serialize + DeserializeOwned,
            A: Agent<AError, EReward, AData>,
            VError: Error,
            V: Visualiser<VError>,
        >(
            count_of_episodes: u128,
        ) -> impl Fn(&E, &A, &V, u128, u128) -> bool {
            move |_environment, _agent, visualiser, episode, _step| {
                !visualiser.is_open() || episode >= count_of_episodes
            }
        }
    }
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- - HELPER - -- -- -- -- -- -- -- -- -- -- -- -- -- -- */

pub fn sleep_suggested_steps_per_second_or_30_fps<E: DrawableEnvironment>() {
    if let Some(rsps) = E::suggested_rendered_steps_per_second() {
        std::thread::sleep(Duration::from_millis((1000f64 / rsps) as u64));
    } else {
        std::thread::sleep(Duration::from_millis((1000f64 / 30f64) as u64));
    }
}

/* -- -- -- -- -- -- -- -- -- -- -- -- --  LOAD AND STORE  -- -- -- -- -- -- -- -- -- -- -- -- -- */

#[derive(Debug)]
enum LoadError<EAError: Error> {
    IoError(std::io::Error),
    SerdeJsonError(serde_json::Error),
    RonError(ron::error::Error),
    BincodeError(Box<bincode::ErrorKind>),
    EnvironmentAgentError(EAError),
    UnknownFormat(String),
}

impl<EAError: Error> Display for LoadError<EAError> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(error) => write!(f, "Received IoError ({})", error),
            Self::SerdeJsonError(error) => write!(f, "Received SerdeJsonError ({})", error),
            Self::RonError(error) => write!(f, "Received RonError ({})", error),
            Self::BincodeError(error) => write!(f, "Received BincodeError ({})", error),
            Self::EnvironmentAgentError(error) => {
                write!(f, "Recedived EnvironmentError({})", error)
            }
            Self::UnknownFormat(path) => {
                write!(f, "The file \"{}\" has an unknown file ending", path)
            }
        }
    }
}

impl<EAError: Error> Error for LoadError<EAError> {}

impl<EAError: Error> From<std::io::Error> for LoadError<EAError> {
    fn from(error: std::io::Error) -> Self {
        Self::IoError(error)
    }
}

impl<EAError: Error> From<serde_json::error::Error> for LoadError<EAError> {
    fn from(error: serde_json::error::Error) -> Self {
        Self::SerdeJsonError(error)
    }
}

impl<EAError: Error> From<Box<bincode::ErrorKind>> for LoadError<EAError> {
    fn from(error: Box<bincode::ErrorKind>) -> Self {
        Self::BincodeError(error)
    }
}

impl<EAError: Error> From<ron::error::Error> for LoadError<EAError> {
    fn from(error: ron::error::Error) -> Self {
        Self::RonError(error)
    }
}

fn load_environment<
    EError: Error,
    EReward: Reward,
    EInfo: Debug,
    EData: Serialize + DeserializeOwned,
    E: Environment<EError, EReward, EInfo, EData>,
>(
    environment: &mut E,
    environment_load_path_string: String,
) -> Result<(), LoadError<EError>> {
    if environment_load_path_string.ends_with(".json") {
        environment
            .load(serde_json::from_reader(std::fs::File::open(
                environment_load_path_string,
            )?)?)
            .map_err(LoadError::EnvironmentAgentError)?;
        Ok(())
    } else if environment_load_path_string.ends_with(".ron") {
        environment
            .load(ron::de::from_reader(std::fs::File::open(
                environment_load_path_string,
            )?)?)
            .map_err(LoadError::EnvironmentAgentError)?;
        Ok(())
    } else if environment_load_path_string.ends_with(".bin") {
        environment
            .load(bincode::deserialize_from(std::fs::File::open(
                environment_load_path_string,
            )?)?)
            .map_err(LoadError::EnvironmentAgentError)?;
        Ok(())
    } else {
        Err(LoadError::UnknownFormat(environment_load_path_string))
    }
}

fn load_agent<
    AError: Error,
    EReward: Reward,
    AData: Serialize + DeserializeOwned,
    A: Agent<AError, EReward, AData>,
>(
    agent: &mut A,
    agent_load_path_string: String,
) -> Result<(), LoadError<AError>> {
    if agent_load_path_string.ends_with(".json") {
        agent
            .load(serde_json::from_reader(std::fs::File::open(
                agent_load_path_string,
            )?)?)
            .map_err(LoadError::EnvironmentAgentError)?;
        Ok(())
    } else if agent_load_path_string.ends_with(".ron") {
        agent
            .load(ron::de::from_reader(std::fs::File::open(
                agent_load_path_string,
            )?)?)
            .map_err(LoadError::EnvironmentAgentError)?;
        Ok(())
    } else if agent_load_path_string.ends_with(".bin") {
        agent
            .load(bincode::deserialize_from(std::fs::File::open(
                agent_load_path_string,
            )?)?)
            .map_err(LoadError::EnvironmentAgentError)?;
        Ok(())
    } else {
        Err(LoadError::UnknownFormat(agent_load_path_string))
    }
}

#[derive(Debug)]
enum StoreError {
    IoError(std::io::Error),
    SerdeJsonError(serde_json::Error),
    RonError(ron::error::Error),
    BincodeError(Box<bincode::ErrorKind>),
    UnknownFormat(String),
}

impl Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(error) => write!(f, "Received IoError ({})", error),
            Self::SerdeJsonError(error) => write!(f, "Received SerdeJsonError ({})", error),
            Self::RonError(error) => write!(f, "Received RonError ({})", error),
            Self::BincodeError(error) => write!(f, "Received BincodeError ({})", error),
            Self::UnknownFormat(path) => {
                write!(f, "The file \"{}\" has an unknown file ending", path)
            }
        }
    }
}

impl Error for StoreError {}

impl From<std::io::Error> for StoreError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError(error)
    }
}

impl From<serde_json::error::Error> for StoreError {
    fn from(error: serde_json::error::Error) -> Self {
        Self::SerdeJsonError(error)
    }
}

impl From<ron::error::Error> for StoreError {
    fn from(error: ron::error::Error) -> Self {
        Self::RonError(error)
    }
}

impl From<Box<bincode::ErrorKind>> for StoreError {
    fn from(error: Box<bincode::ErrorKind>) -> Self {
        Self::BincodeError(error)
    }
}

fn store_environment<
    EError: Error,
    EReward: Reward,
    EInfo: Debug,
    EData: Serialize + DeserializeOwned,
    E: Environment<EError, EReward, EInfo, EData>,
>(
    environment: &E,
    environment_store_path_string: String,
) -> Result<(), StoreError> {
    if environment_store_path_string.ends_with(".json") {
        serde_json::to_writer(
            std::fs::File::create(environment_store_path_string)?,
            &environment.store(),
        )?;
        Ok(())
    } else if environment_store_path_string.ends_with(".ron") {
        ron::ser::to_writer(
            std::fs::File::create(environment_store_path_string)?,
            &environment.store(),
        )?;
        Ok(())
    } else if environment_store_path_string.ends_with(".bin") {
        bincode::serialize_into(
            std::fs::File::create(environment_store_path_string)?,
            &environment.store(),
        )?;
        Ok(())
    } else {
        Err(StoreError::UnknownFormat(environment_store_path_string))
    }
}

fn store_agent<
    AError: Error,
    EReward: Reward,
    AData: Serialize + DeserializeOwned,
    A: Agent<AError, EReward, AData>,
>(
    agent: &A,
    agent_store_path_string: String,
) -> Result<(), StoreError> {
    if agent_store_path_string.ends_with(".json") {
        serde_json::to_writer(
            std::fs::File::create(agent_store_path_string)?,
            &agent.store(),
        )?;
        Ok(())
    } else if agent_store_path_string.ends_with(".ron") {
        ron::ser::to_writer(
            std::fs::File::create(agent_store_path_string)?,
            &agent.store(),
        )?;
        Ok(())
    } else if agent_store_path_string.ends_with(".bin") {
        bincode::serialize_into(
            std::fs::File::create(agent_store_path_string)?,
            &agent.store(),
        )?;
        Ok(())
    } else {
        Err(StoreError::UnknownFormat(agent_store_path_string))
    }
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- */
